# Telegram VS Code Bridge Extension

Extensión mínima para exponer un bridge HTTP local y ejecutar comandos de VS Code desde procesos locales (por ejemplo, un bot de Telegram).

## Qué hace

- Levanta un servidor HTTP local (`http.createServer`) en `telegramBridge.host:telegramBridge.port`.
- Exige token compartido por header `X-Bridge-Token` o `Authorization: Bearer <token>`.
- Expone:
  - `GET /health`
  - `GET /commands`
  - `POST /command`
- Ejecuta únicamente comandos dentro de `telegramBridge.allowedCommands`.
- Incluye comando interno:
  - `telegramBridge.showLogs`
  - `telegramBridge.openAndAddFile` (abre archivo y luego intenta `chatgpt.addFileToThread`).

## Settings

- `telegramBridge.host` (default `127.0.0.1`)
- `telegramBridge.port` (default `8787`)
- `telegramBridge.token` (sin default seguro; debe configurarse)
- `telegramBridge.allowedCommands` (array de command IDs permitidos)
- `telegramBridge.enableVerboseLogs` (bool)

Si cambia cualquier setting de `telegramBridge.*`, la extensión reinicia automáticamente el servidor.

## Desarrollo y ejecución con F5

1. Abrí esta carpeta (`vscode-telegram-bridge-extension`) en VS Code.
2. Instalá dependencias:
   - `npm install`
3. Presioná `F5` para lanzar **Extension Development Host**.
4. En la ventana del host, configurá settings por `settings.json`:

```json
{
  "telegramBridge.host": "127.0.0.1",
  "telegramBridge.port": 8787,
  "telegramBridge.token": "cambia-este-token",
  "telegramBridge.allowedCommands": [
    "chatgpt.openSidebar",
    "chatgpt.newChat",
    "chatgpt.addFileToThread",
    "telegramBridge.openAndAddFile"
  ],
  "telegramBridge.enableVerboseLogs": true
}
```

5. Abrí `Command Palette` y ejecutá `Telegram Bridge: Show Logs` para ver logs.

## API del bridge

Todos los requests requieren token.

### `GET /health`

Respuesta ejemplo:

```json
{
  "ok": true,
  "name": "telegram-vscode-bridge",
  "host": "127.0.0.1",
  "port": 8787,
  "tokenConfigured": true,
  "allowedCommandsCount": 4,
  "uptimeSec": 52,
  "extensionVersion": "0.0.1"
}
```

### `GET /commands`

Respuesta ejemplo:

```json
{
  "ok": true,
  "allowedCommands": [
    "chatgpt.openSidebar",
    "chatgpt.newChat",
    "chatgpt.addFileToThread",
    "telegramBridge.openAndAddFile"
  ]
}
```

### `POST /command`

Body:

```json
{
  "command": "chatgpt.openSidebar",
  "args": []
}
```

Respuesta ejemplo:

```json
{
  "ok": true,
  "command": "chatgpt.openSidebar",
  "durationMs": 35,
  "result": null
}
```

Si el comando no está en allowlist:

```json
{
  "ok": false,
  "error": "Command \"...\" is not in telegramBridge.allowedCommands."
}
```
