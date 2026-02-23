import * as crypto from 'crypto';
import * as fs from 'fs';
import * as http from 'http';
import * as os from 'os';
import * as path from 'path';
import * as vscode from 'vscode';

const CONFIG_SECTION = 'telegramBridge';
const MAX_BODY_BYTES = 64 * 1024;
const LOCALHOST_FALLBACK = '127.0.0.1';

interface BridgeConfig {
  host: string;
  port: number;
  token: string;
  allowedCommands: string[];
  enableVerboseLogs: boolean;
}

interface CommandRequestPayload {
  command: string;
  args: unknown[];
}

class TelegramBridgeController implements vscode.Disposable {
  private readonly context: vscode.ExtensionContext;
  private readonly output: vscode.OutputChannel;
  private readonly disposables: vscode.Disposable[] = [];
  private server?: http.Server;
  private config: BridgeConfig;
  private readonly startedAtMs = Date.now();

  constructor(context: vscode.ExtensionContext) {
    this.context = context;
    this.output = vscode.window.createOutputChannel('Telegram Bridge');
    this.config = this.loadConfig();

    this.disposables.push(
      this.output,
      vscode.commands.registerCommand('telegramBridge.showLogs', () => {
        this.output.show(true);
      }),
      vscode.commands.registerCommand('telegramBridge.openAndAddFile', async (inputPath: unknown) => {
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

  dispose(): void {
    void this.stopServer();
    for (const disposable of this.disposables) {
      disposable.dispose();
    }
  }

  private loadConfig(): BridgeConfig {
    const config = vscode.workspace.getConfiguration(CONFIG_SECTION);

    const hostInput = String(config.get<string>('host', LOCALHOST_FALLBACK)).trim();
    const host = this.normalizeHost(hostInput);

    const rawPort = Number(config.get<number>('port', 8787));
    const port = Number.isInteger(rawPort) && rawPort > 0 && rawPort <= 65535 ? rawPort : 8787;

    const token = String(config.get<string>('token', '')).trim();
    const allowedCommands = this.normalizeAllowedCommands(config.get<unknown[]>('allowedCommands', []));
    const enableVerboseLogs = Boolean(config.get<boolean>('enableVerboseLogs', false));

    return {
      host,
      port,
      token,
      allowedCommands,
      enableVerboseLogs
    };
  }

  private normalizeHost(host: string): string {
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

  private normalizeAllowedCommands(value: unknown[]): string[] {
    const uniqueCommands = new Set<string>();
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

  private async reloadConfig(): Promise<void> {
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

  private async restartServer(): Promise<void> {
    await this.stopServer();
    await this.startServer();
  }

  private async startServer(): Promise<void> {
    if (this.server) {
      return;
    }

    this.server = http.createServer((req, res) => {
      void this.handleHttpRequest(req, res);
    });

    await new Promise<void>((resolve, reject) => {
      if (!this.server) {
        reject(new Error('Bridge server instance disappeared during startup.'));
        return;
      }

      const onError = (error: Error): void => {
        this.server?.off('listening', onListening);
        reject(error);
      };

      const onListening = (): void => {
        this.server?.off('error', onError);
        resolve();
      };

      this.server.once('error', onError);
      this.server.once('listening', onListening);
      this.server.listen(this.config.port, this.config.host);
    }).catch((error: unknown) => {
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

  private async stopServer(): Promise<void> {
    if (!this.server) {
      return;
    }

    const currentServer = this.server;
    this.server = undefined;

    await new Promise<void>((resolve) => {
      currentServer.close((error) => {
        if (error) {
          this.log(`Bridge close error: ${this.formatError(error)}`, false);
        }
        resolve();
      });
    });

    this.log('Bridge server stopped.', true);
  }

  private async handleHttpRequest(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const method = (req.method ?? 'GET').toUpperCase();

    let parsedUrl: URL;
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

  private async handleCommandRequest(
    req: http.IncomingMessage,
    res: http.ServerResponse
  ): Promise<void> {
    let payloadRaw: unknown;
    try {
      payloadRaw = await this.readJsonBody(req);
    } catch (error: unknown) {
      const message = this.formatError(error);
      this.respondJson(res, 400, { ok: false, error: `Invalid JSON body: ${message}` });
      return;
    }

    let payload: CommandRequestPayload;
    try {
      payload = this.parseCommandPayload(payloadRaw);
    } catch (error: unknown) {
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
    } catch (error: unknown) {
      const message = this.formatError(error);
      this.respondJson(res, 500, {
        ok: false,
        command: payload.command,
        error: message
      });
      this.log(`Command "${payload.command}" failed: ${message}`, false);
    }
  }

  private async readJsonBody(req: http.IncomingMessage): Promise<unknown> {
    return new Promise<unknown>((resolve, reject) => {
      const chunks: Buffer[] = [];
      let bodyLength = 0;

      req.on('data', (chunk: Buffer) => {
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

  private parseCommandPayload(value: unknown): CommandRequestPayload {
    if (!value || typeof value !== 'object') {
      throw new Error('Body must be a JSON object.');
    }

    const rawCommand = (value as { command?: unknown }).command;
    const rawArgs = (value as { args?: unknown }).args;

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

  private isAuthorized(req: http.IncomingMessage): boolean {
    const tokenFromRequest = this.getTokenFromRequest(req);
    if (!tokenFromRequest) {
      return false;
    }
    return this.tokensEqual(tokenFromRequest, this.config.token);
  }

  private getTokenFromRequest(req: http.IncomingMessage): string | undefined {
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

  private tokensEqual(left: string, right: string): boolean {
    const leftBuffer = Buffer.from(left, 'utf8');
    const rightBuffer = Buffer.from(right, 'utf8');
    if (leftBuffer.length !== rightBuffer.length) {
      return false;
    }
    return crypto.timingSafeEqual(leftBuffer, rightBuffer);
  }

  private respondJson(res: http.ServerResponse, statusCode: number, payload: Record<string, unknown>): void {
    const body = JSON.stringify(payload);
    res.statusCode = statusCode;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.setHeader('Content-Length', Buffer.byteLength(body));
    res.end(body);
  }

  private toSerializable(value: unknown): unknown {
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

  private log(message: string, verboseOnly: boolean): void {
    if (verboseOnly && !this.config.enableVerboseLogs) {
      return;
    }
    this.output.appendLine(`[${new Date().toISOString()}] ${message}`);
  }

  private formatError(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }
    return String(error);
  }

  private async openAndAddFile(inputPath: string): Promise<Record<string, unknown>> {
    const fileUri = this.resolveInputPath(inputPath);

    await vscode.commands.executeCommand('vscode.open', fileUri);

    try {
      await vscode.commands.executeCommand('chatgpt.addFileToThread', fileUri);
      return {
        opened: fileUri.fsPath,
        addedToThread: true,
        mode: 'chatgpt.addFileToThread(fileUri)'
      };
    } catch (errorWithUri: unknown) {
      try {
        await vscode.commands.executeCommand('chatgpt.addFileToThread');
        return {
          opened: fileUri.fsPath,
          addedToThread: true,
          mode: 'chatgpt.addFileToThread()',
          note:
            'chatgpt.addFileToThread(fileUri) failed, fallback without args worked.'
        };
      } catch (fallbackError: unknown) {
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

  private resolveInputPath(inputPath: string): vscode.Uri {
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

  private expandHome(inputPath: string): string {
    if (inputPath === '~') {
      return os.homedir();
    }
    if (inputPath.startsWith('~/') || inputPath.startsWith('~\\')) {
      return path.join(os.homedir(), inputPath.slice(2));
    }
    return inputPath;
  }

  private resolveAbsolutePath(inputPath: string): string {
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

export function activate(context: vscode.ExtensionContext): void {
  const controller = new TelegramBridgeController(context);
  context.subscriptions.push(controller);
}

export function deactivate(): void {
  // No-op. Resources are disposed via context subscriptions.
}
