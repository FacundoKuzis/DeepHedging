'use strict';

const crypto = require('crypto');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');
const vscode = require('vscode');

const CONFIG_SECTION = 'telegramBridge';
const MAX_BODY_BYTES = 64 * 1024;
const LOCALHOST_FALLBACK = '127.0.0.1';

class TelegramBridgeController {
  constructor(context) {
    this.context = context;
    this.output = vscode.window.createOutputChannel('Telegram Bridge');
    this.disposables = [];
    this.server = undefined;
    this.config = this.loadConfig();
    this.startedAtMs = Date.now();

    this.disposables.push(
      this.output,
      vscode.commands.registerCommand('telegramBridge.showLogs', () => {
        this.output.show(true);
      }),
      vscode.commands.registerCommand('telegramBridge.openAndAddFile', async (inputPath) => {
        if (typeof inputPath !== 'string' || inputPath.trim() === '') {
          throw new Error('telegramBridge.openAndAddFile requires one non-empty path argument.');
        }
        return this.openAndAddFile(inputPath);
      }),
      vscode.workspace.onDidChangeConfiguration((event) => {
        if (!event.affectsConfiguration(CONFIG_SECTION)) {
          return;
        }
        void this.reloadConfig();
      })
    );

    void this.startServer();
  }

  dispose() {
    void this.stopServer();
    for (const disposable of this.disposables) {
      disposable.dispose();
    }
  }

  loadConfig() {
    const config = vscode.workspace.getConfiguration(CONFIG_SECTION);

    const hostInput = String(config.get('host', LOCALHOST_FALLBACK)).trim();
    const host = this.normalizeHost(hostInput);

    const rawPort = Number(config.get('port', 8787));
    const port = Number.isInteger(rawPort) && rawPort > 0 && rawPort <= 65535 ? rawPort : 8787;

    const token = String(config.get('token', '')).trim();
    const allowedCommands = this.normalizeAllowedCommands(config.get('allowedCommands', []));
    const enableVerboseLogs = Boolean(config.get('enableVerboseLogs', false));

    return {
      host,
      port,
      token,
      allowedCommands,
      enableVerboseLogs
    };
  }

  normalizeHost(host) {
    const normalized = host.toLowerCase();
    const allowedHosts = new Set(['127.0.0.1', 'localhost', '::1']);
    if (allowedHosts.has(normalized)) {
      return host;
    }

    this.log(
      `Configured host "${host}" is not localhost. Falling back to ${LOCALHOST_FALLBACK}.`,
      false
    );
    return LOCALHOST_FALLBACK;
  }

  normalizeAllowedCommands(value) {
    const uniqueCommands = new Set();
    for (const item of value) {
      if (typeof item !== 'string') {
        continue;
      }
      const commandId = item.trim();
      if (commandId.length > 0) {
        uniqueCommands.add(commandId);
      }
    }
    return Array.from(uniqueCommands);
  }

  async reloadConfig() {
    const previous = this.config;
    this.config = this.loadConfig();

    const changed =
      previous.host !== this.config.host ||
      previous.port !== this.config.port ||
      previous.token !== this.config.token ||
      previous.enableVerboseLogs !== this.config.enableVerboseLogs ||
      previous.allowedCommands.join('\n') !== this.config.allowedCommands.join('\n');

    if (!changed) {
      return;
    }

    this.log('Detected telegramBridge configuration change. Restarting bridge server.', false);
    await this.restartServer();
  }

  async restartServer() {
    await this.stopServer();
    await this.startServer();
  }

  async startServer() {
    if (this.server) {
      return;
    }

    this.server = http.createServer((req, res) => {
      void this.handleHttpRequest(req, res);
    });

    await new Promise((resolve, reject) => {
      if (!this.server) {
        reject(new Error('Bridge server instance disappeared during startup.'));
        return;
      }

      const onError = (error) => {
        this.server?.off('listening', onListening);
        reject(error);
      };

      const onListening = () => {
        this.server?.off('error', onError);
        resolve();
      };

      this.server.once('error', onError);
      this.server.once('listening', onListening);
      this.server.listen(this.config.port, this.config.host);
    }).catch((error) => {
      this.server = undefined;
      this.log(`Bridge failed to start: ${this.formatError(error)}`, false);
      throw error;
    });

    this.log(
      `Bridge listening on http://${this.config.host}:${this.config.port}. Allowed commands: ${this.config.allowedCommands.length}.`,
      false
    );

    if (!this.config.token) {
      this.log(
        'telegramBridge.token is empty. All bridge requests will return 503 until token is configured.',
        false
      );
    }
  }

  async stopServer() {
    if (!this.server) {
      return;
    }

    const currentServer = this.server;
    this.server = undefined;

    await new Promise((resolve) => {
      currentServer.close((error) => {
        if (error) {
          this.log(`Bridge close error: ${this.formatError(error)}`, false);
        }
        resolve();
      });
    });

    this.log('Bridge server stopped.', true);
  }

  async handleHttpRequest(req, res) {
    const method = (req.method ?? 'GET').toUpperCase();

    let parsedUrl;
    try {
      parsedUrl = new URL(req.url ?? '/', `http://${this.config.host}:${this.config.port}`);
    } catch {
      this.respondJson(res, 400, { ok: false, error: 'Invalid request URL.' });
      return;
    }

    this.log(`${method} ${parsedUrl.pathname}`, true);

    if (!this.config.token) {
      this.respondJson(res, 503, {
        ok: false,
        error: 'Bridge token is not configured in telegramBridge.token.'
      });
      return;
    }

    if (!this.isAuthorized(req)) {
      this.respondJson(res, 401, { ok: false, error: 'Unauthorized.' });
      return;
    }

    if (method === 'GET' && parsedUrl.pathname === '/health') {
      this.respondJson(res, 200, {
        ok: true,
        name: 'telegram-vscode-bridge',
        host: this.config.host,
        port: this.config.port,
        tokenConfigured: this.config.token.length > 0,
        allowedCommandsCount: this.config.allowedCommands.length,
        uptimeSec: Math.floor((Date.now() - this.startedAtMs) / 1000),
        extensionVersion: this.context.extension.packageJSON.version
      });
      return;
    }

    if (method === 'GET' && parsedUrl.pathname === '/commands') {
      this.respondJson(res, 200, {
        ok: true,
        allowedCommands: this.config.allowedCommands
      });
      return;
    }

    if (method === 'POST' && parsedUrl.pathname === '/command') {
      await this.handleCommandRequest(req, res);
      return;
    }

    this.respondJson(res, 404, { ok: false, error: 'Not found.' });
  }

  async handleCommandRequest(req, res) {
    let payloadRaw;
    try {
      payloadRaw = await this.readJsonBody(req);
    } catch (error) {
      const message = this.formatError(error);
      this.respondJson(res, 400, { ok: false, error: `Invalid JSON body: ${message}` });
      return;
    }

    let payload;
    try {
      payload = this.parseCommandPayload(payloadRaw);
    } catch (error) {
      this.respondJson(res, 400, { ok: false, error: this.formatError(error) });
      return;
    }

    if (!this.config.allowedCommands.includes(payload.command)) {
      this.respondJson(res, 403, {
        ok: false,
        error: `Command "${payload.command}" is not in telegramBridge.allowedCommands.`
      });
      return;
    }

    const started = Date.now();
    try {
      const result = await vscode.commands.executeCommand(payload.command, ...payload.args);
      this.respondJson(res, 200, {
        ok: true,
        command: payload.command,
        durationMs: Date.now() - started,
        result: this.toSerializable(result)
      });
      this.log(`Executed command "${payload.command}".`, false);
    } catch (error) {
      const message = this.formatError(error);
      this.respondJson(res, 500, {
        ok: false,
        command: payload.command,
        error: message
      });
      this.log(`Command "${payload.command}" failed: ${message}`, false);
    }
  }

  async readJsonBody(req) {
    return new Promise((resolve, reject) => {
      const chunks = [];
      let bodyLength = 0;

      req.on('data', (chunk) => {
        bodyLength += chunk.length;
        if (bodyLength > MAX_BODY_BYTES) {
          reject(new Error(`Body too large. Max size is ${MAX_BODY_BYTES} bytes.`));
          req.destroy();
          return;
        }
        chunks.push(chunk);
      });

      req.on('error', (error) => {
        reject(error);
      });

      req.on('end', () => {
        if (chunks.length === 0) {
          resolve({});
          return;
        }

        const text = Buffer.concat(chunks).toString('utf8');
        try {
          resolve(JSON.parse(text));
        } catch (error) {
          reject(error);
        }
      });
    });
  }

  parseCommandPayload(value) {
    if (!value || typeof value !== 'object') {
      throw new Error('Body must be a JSON object.');
    }

    const rawCommand = value.command;
    const rawArgs = value.args;

    if (typeof rawCommand !== 'string' || rawCommand.trim() === '') {
      throw new Error('"command" must be a non-empty string.');
    }

    if (rawArgs !== undefined && !Array.isArray(rawArgs)) {
      throw new Error('"args" must be an array if provided.');
    }

    return {
      command: rawCommand.trim(),
      args: Array.isArray(rawArgs) ? rawArgs : []
    };
  }

  isAuthorized(req) {
    const tokenFromRequest = this.getTokenFromRequest(req);
    if (!tokenFromRequest) {
      return false;
    }
    return this.tokensEqual(tokenFromRequest, this.config.token);
  }

  getTokenFromRequest(req) {
    const tokenHeader = req.headers['x-bridge-token'];
    const tokenValue = Array.isArray(tokenHeader) ? tokenHeader[0] : tokenHeader;
    if (typeof tokenValue === 'string' && tokenValue.trim()) {
      return tokenValue.trim();
    }

    const authHeader = req.headers.authorization;
    const authValue = Array.isArray(authHeader) ? authHeader[0] : authHeader;
    if (typeof authValue === 'string') {
      const match = authValue.match(/^Bearer\s+(.+)$/i);
      if (match && match[1].trim()) {
        return match[1].trim();
      }
    }

    return undefined;
  }

  tokensEqual(left, right) {
    const leftBuffer = Buffer.from(left, 'utf8');
    const rightBuffer = Buffer.from(right, 'utf8');
    if (leftBuffer.length !== rightBuffer.length) {
      return false;
    }
    return crypto.timingSafeEqual(leftBuffer, rightBuffer);
  }

  respondJson(res, statusCode, payload) {
    const body = JSON.stringify(payload);
    res.statusCode = statusCode;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.setHeader('Content-Length', Buffer.byteLength(body));
    res.end(body);
  }

  toSerializable(value) {
    if (value === undefined) {
      return null;
    }
    if (value === null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      return value;
    }
    try {
      return JSON.parse(JSON.stringify(value));
    } catch {
      return String(value);
    }
  }

  log(message, verboseOnly) {
    if (verboseOnly && !this.config.enableVerboseLogs) {
      return;
    }
    this.output.appendLine(`[${new Date().toISOString()}] ${message}`);
  }

  formatError(error) {
    if (error instanceof Error) {
      return error.message;
    }
    return String(error);
  }

  async openAndAddFile(inputPath) {
    const fileUri = this.resolveInputPath(inputPath);

    await vscode.commands.executeCommand('vscode.open', fileUri);

    try {
      await vscode.commands.executeCommand('chatgpt.addFileToThread', fileUri);
      return {
        opened: fileUri.fsPath,
        addedToThread: true,
        mode: 'chatgpt.addFileToThread(fileUri)'
      };
    } catch (errorWithUri) {
      try {
        await vscode.commands.executeCommand('chatgpt.addFileToThread');
        return {
          opened: fileUri.fsPath,
          addedToThread: true,
          mode: 'chatgpt.addFileToThread()',
          note:
            'chatgpt.addFileToThread(fileUri) failed, fallback without args worked.'
        };
      } catch (fallbackError) {
        throw new Error(
          [
            `Opened "${fileUri.fsPath}" but failed to add it to Codex thread.`,
            `With URI: ${this.formatError(errorWithUri)}`,
            `Without args: ${this.formatError(fallbackError)}`
          ].join(' ')
        );
      }
    }
  }

  resolveInputPath(inputPath) {
    const expandedInput = this.expandHome(inputPath.trim());
    const absolutePath = this.resolveAbsolutePath(expandedInput);

    if (!fs.existsSync(absolutePath)) {
      throw new Error(`File does not exist: ${absolutePath}`);
    }

    const stat = fs.statSync(absolutePath);
    if (!stat.isFile()) {
      throw new Error(`Path is not a file: ${absolutePath}`);
    }

    return vscode.Uri.file(absolutePath);
  }

  expandHome(inputPath) {
    if (inputPath === '~') {
      return os.homedir();
    }
    if (inputPath.startsWith('~/') || inputPath.startsWith('~\\')) {
      return path.join(os.homedir(), inputPath.slice(2));
    }
    return inputPath;
  }

  resolveAbsolutePath(inputPath) {
    if (path.isAbsolute(inputPath)) {
      return path.normalize(inputPath);
    }

    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (workspaceFolder) {
      return path.normalize(path.join(workspaceFolder.uri.fsPath, inputPath));
    }

    return path.normalize(path.resolve(inputPath));
  }
}

function activate(context) {
  const controller = new TelegramBridgeController(context);
  context.subscriptions.push(controller);
}

function deactivate() {
  // No-op. Resources are disposed via context subscriptions.
}

module.exports = {
  activate,
  deactivate
};
