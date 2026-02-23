from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# ==========================
# Global configuration
# ==========================
TELEGRAM_BOT_TOKEN = "8623040929:AAF5NPNKFHg_u6EAJibAOe1QzdQFIQc_1DQ"
ALLOWED_CHAT_IDS: Set[int] = set()
VSCODE_BRIDGE_URL = "http://127.0.0.1:8787"
VSCODE_BRIDGE_TOKEN = "KACPFxxkpHUPsJ96o5mMeqbm3LO0rvhoPt4Oxym5bZ4"
REQUEST_TIMEOUT_SEC = 10

# Optional tuning knobs
LONG_POLL_TIMEOUT_SEC = 30
RATE_LIMIT_SECONDS = 1.0
POLL_RETRY_DELAY_SEC = 2
ADDFILE_BRIDGE_COMMAND = "telegramBridge.openAndAddFile"
STATE_FILE_PATH = ".telegram_bridge_state.json"
AUTO_ALLOW_FIRST_CHAT = True
DELETE_WEBHOOK_ON_START = True

DEFAULT_COMMAND_MAP: Dict[str, str] = {
    "/sidebar": "chatgpt.openSidebar",
    "/newchat": "chatgpt.newChat",
}


class BridgeClient:
    def __init__(self, base_url: str, token: str, timeout_sec: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_sec = timeout_sec
        self.session = requests.Session()

    def health(self) -> Tuple[bool, Any]:
        return self._request("GET", "/health")

    def list_commands(self) -> Tuple[bool, Any]:
        return self._request("GET", "/commands")

    def run_command(self, command_id: str, args: Optional[List[Any]] = None) -> Tuple[bool, Any]:
        payload = {
            "command": command_id,
            "args": args if args is not None else [],
        }
        return self._request("POST", "/command", payload=payload)

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "X-Bridge-Token": self.token,
        }

        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, timeout=self.timeout_sec)
            elif method == "POST":
                response = self.session.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
            else:
                return False, f"Metodo HTTP no soportado: {method}"
        except requests.RequestException as exc:
            return False, f"Error de red hacia bridge: {exc}"

        try:
            data = response.json()
        except ValueError:
            snippet = response.text[:200].replace("\n", " ")
            return (
                False,
                f"Respuesta no-JSON del bridge (HTTP {response.status_code}): {snippet}",
            )

        if not isinstance(data, dict):
            return False, f"Respuesta JSON invalida del bridge (se esperaba objeto): {data!r}"

        if response.status_code >= 400:
            error_message = data.get("error", f"HTTP {response.status_code}")
            return False, f"Bridge devolvio error HTTP {response.status_code}: {error_message}"

        return True, data


class TelegramVsCodeBridgeBot:
    def __init__(
        self,
        telegram_token: str,
        allowed_chat_ids: Set[int],
        bridge_client: BridgeClient,
    ) -> None:
        self.telegram_token = telegram_token
        self.allowed_chat_ids = set(allowed_chat_ids)
        self.bridge_client = bridge_client
        self.telegram_session = requests.Session()
        self.update_offset: Optional[int] = None
        self.last_command_ts: Dict[int, float] = {}
        self.state_file = Path(STATE_FILE_PATH)
        self._load_state()

    def run_forever(self) -> None:
        if DELETE_WEBHOOK_ON_START:
            self._disable_webhook_for_long_polling()
        self._log("Bot iniciado. Esperando mensajes...")
        while True:
            ok, updates_or_error = self._get_updates()
            if not ok:
                self._log(f"Fallo getUpdates: {updates_or_error}")
                time.sleep(POLL_RETRY_DELAY_SEC)
                continue

            updates = updates_or_error
            for update in updates:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    self.update_offset = update_id + 1

                message = update.get("message")
                if isinstance(message, dict):
                    self._handle_message(message)

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return

        try:
            raw = self.state_file.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as exc:
            self._log(f"No se pudo cargar {self.state_file}: {exc}")
            return

        persisted_ids = data.get("allowed_chat_ids", [])
        if not isinstance(persisted_ids, list):
            return

        for value in persisted_ids:
            if isinstance(value, int):
                self.allowed_chat_ids.add(value)

        if self.allowed_chat_ids:
            self._log(f"Chats autorizados cargados desde disco: {sorted(self.allowed_chat_ids)}")

    def _save_state(self) -> None:
        payload = {
            "allowed_chat_ids": sorted(self.allowed_chat_ids),
            "updated_at_epoch": int(time.time()),
        }
        try:
            self.state_file.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            self._log(f"No se pudo guardar estado en {self.state_file}: {exc}")

    def _disable_webhook_for_long_polling(self) -> None:
        ok, result_or_error = self._telegram_api(
            "deleteWebhook",
            {"drop_pending_updates": False},
            timeout_sec=REQUEST_TIMEOUT_SEC,
        )
        if ok:
            self._log("Webhook deshabilitado para usar long polling.")
        else:
            self._log(f"No se pudo deshabilitar webhook: {result_or_error}")

    def _get_updates(self) -> Tuple[bool, Any]:
        payload: Dict[str, Any] = {
            "timeout": LONG_POLL_TIMEOUT_SEC,
            "allowed_updates": ["message"],
        }
        if self.update_offset is not None:
            payload["offset"] = self.update_offset

        return self._telegram_api(
            "getUpdates",
            payload,
            timeout_sec=LONG_POLL_TIMEOUT_SEC + REQUEST_TIMEOUT_SEC,
        )

    def _telegram_api(
        self,
        method: str,
        payload: Dict[str, Any],
        timeout_sec: int,
    ) -> Tuple[bool, Any]:
        url = f"https://api.telegram.org/bot{self.telegram_token}/{method}"
        try:
            response = self.telegram_session.post(url, json=payload, timeout=timeout_sec)
        except requests.RequestException as exc:
            return False, f"Error de red con Telegram ({method}): {exc}"

        try:
            data = response.json()
        except ValueError:
            snippet = response.text[:200].replace("\n", " ")
            return False, f"Respuesta invalida de Telegram ({method}): {snippet}"

        if response.status_code >= 400:
            return False, f"Telegram HTTP {response.status_code}: {data}"

        if not isinstance(data, dict):
            return False, f"Respuesta JSON inesperada de Telegram ({method}): {data!r}"

        if not data.get("ok"):
            return False, f"Telegram API reporto ok=false ({method}): {data}"

        return True, data.get("result", [])

    def _handle_message(self, message: Dict[str, Any]) -> None:
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        if not isinstance(chat_id, int):
            return

        text = message.get("text")
        if not isinstance(text, str):
            return

        text = text.strip()
        if not text.startswith("/"):
            return

        command, tail = self._split_command(text)
        command = self._normalize_command(command)

        if chat_id not in self.allowed_chat_ids:
            if self._try_auto_allow(chat_id, command):
                self._send_message(
                    chat_id,
                    (
                        "Chat autorizado para controlar VS Code.\n"
                        "Usa /help para ver comandos."
                    ),
                )
                self._log(f"Chat auto-autorizado: {chat_id}")
            else:
                self._log(f"Mensaje ignorado de chat no autorizado: {chat_id}")
            return

        if command in {"/start", "/help", "/myid"}:
            self._handle_non_rate_limited(chat_id, command)
            return

        allowed_now, wait_seconds = self._check_rate_limit(chat_id)
        if not allowed_now:
            self._send_message(
                chat_id,
                f"Rate limit activo. Espera {wait_seconds:.1f}s antes del siguiente comando.",
            )
            return

        if command == "/health":
            self._handle_health(chat_id)
            return
        if command == "/sidebar":
            self._execute_bridge_command(chat_id, DEFAULT_COMMAND_MAP["/sidebar"], [])
            return
        if command == "/newchat":
            self._execute_bridge_command(chat_id, DEFAULT_COMMAND_MAP["/newchat"], [])
            return
        if command == "/addfile":
            self._handle_add_file(chat_id, tail)
            return
        if command == "/runcommand":
            self._handle_runcommand(chat_id, tail)
            return
        if command == "/listdefaults":
            self._handle_listdefaults(chat_id)
            return
        if command == "/commands":
            self._handle_commands(chat_id)
            return

        self._send_message(
            chat_id,
            "Comando no reconocido. Usa /help para ver opciones disponibles.",
        )

    def _try_auto_allow(self, chat_id: int, command: str) -> bool:
        if not AUTO_ALLOW_FIRST_CHAT:
            return False
        if self.allowed_chat_ids:
            return False
        if command != "/start":
            return False
        self.allowed_chat_ids.add(chat_id)
        self._save_state()
        return True

    def _handle_non_rate_limited(self, chat_id: int, command: str) -> None:
        if command == "/start":
            self._send_message(
                chat_id,
                "Bridge bot activo.\nUsa /help para ver comandos.",
            )
            return
        if command == "/myid":
            self._send_message(chat_id, f"Tu chat_id es: {chat_id}")
            return
        self._send_help(chat_id)

    def _handle_health(self, chat_id: int) -> None:
        ok, data_or_error = self.bridge_client.health()
        if not ok:
            self._send_message(chat_id, f"Health error: {data_or_error}")
            return

        data = data_or_error
        text = (
            "Bridge OK\n"
            f"host: {data.get('host')}:{data.get('port')}\n"
            f"allowlist: {data.get('allowedCommandsCount')}\n"
            f"tokenConfigured: {data.get('tokenConfigured')}\n"
            f"uptimeSec: {data.get('uptimeSec')}"
        )
        self._send_message(chat_id, text)

    def _handle_add_file(self, chat_id: int, tail: str) -> None:
        file_path = tail.strip()
        if not file_path:
            self._send_message(chat_id, "Uso: /addfile <ruta>")
            return

        self._execute_bridge_command(chat_id, ADDFILE_BRIDGE_COMMAND, [file_path])

    def _handle_runcommand(self, chat_id: int, tail: str) -> None:
        if not tail.strip():
            self._send_message(chat_id, "Uso: /runcommand <commandId>")
            return

        command_id = tail.strip().split()[0]
        self._execute_bridge_command(chat_id, command_id, [])

    def _handle_listdefaults(self, chat_id: int) -> None:
        lines = [
            "Mapa por defecto:",
            f"/sidebar -> {DEFAULT_COMMAND_MAP['/sidebar']}",
            f"/newchat -> {DEFAULT_COMMAND_MAP['/newchat']}",
            f"/addfile <ruta> -> {ADDFILE_BRIDGE_COMMAND}",
            "/runcommand <commandId> -> (comando directo del bridge)",
            "/commands -> lista allowlist actual del bridge",
        ]
        self._send_message(chat_id, "\n".join(lines))

    def _handle_commands(self, chat_id: int) -> None:
        ok, data_or_error = self.bridge_client.list_commands()
        if not ok:
            self._send_message(chat_id, f"Error consultando /commands: {data_or_error}")
            return

        commands = data_or_error.get("allowedCommands")
        if not isinstance(commands, list):
            self._send_message(chat_id, "Respuesta invalida en /commands (allowedCommands faltante).")
            return

        if not commands:
            self._send_message(chat_id, "Allowlist vacia en bridge.")
            return

        text = "Allowlist del bridge:\n" + "\n".join(f"- {cmd}" for cmd in commands)
        self._send_message(chat_id, text)

    def _execute_bridge_command(self, chat_id: int, command_id: str, args: List[Any]) -> None:
        ok, data_or_error = self.bridge_client.run_command(command_id, args=args)
        if not ok:
            self._send_message(chat_id, f"Error ejecutando {command_id}: {data_or_error}")
            return

        result = data_or_error
        if not result.get("ok"):
            self._send_message(chat_id, f"Bridge devolvio ok=false para {command_id}: {result}")
            return

        duration_ms = result.get("durationMs")
        command_result = result.get("result")
        lines = [
            f"OK: {command_id}",
            f"durationMs: {duration_ms}",
        ]
        if command_result not in (None, ""):
            lines.append(f"result: {command_result}")
        self._send_message(chat_id, "\n".join(lines))

    def _send_help(self, chat_id: int) -> None:
        help_text = (
            "Comandos disponibles:\n"
            "/start\n"
            "/help\n"
            "/health\n"
            "/sidebar\n"
            "/newchat\n"
            "/addfile <ruta>\n"
            "/runcommand <commandId>\n"
            "/listdefaults\n"
            "/commands\n"
            "/myid"
        )
        self._send_message(chat_id, help_text)

    def _send_message(self, chat_id: int, text: str) -> None:
        payload = {
            "chat_id": chat_id,
            "text": text,
        }
        ok, result = self._telegram_api("sendMessage", payload, timeout_sec=REQUEST_TIMEOUT_SEC)
        if not ok:
            self._log(f"Fallo enviando mensaje a {chat_id}: {result}")

    def _split_command(self, text: str) -> Tuple[str, str]:
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[1]

    def _normalize_command(self, command: str) -> str:
        # Telegram may send "/command@BotName" in groups.
        if "@" in command:
            command = command.split("@", 1)[0]
        return command.lower()

    def _check_rate_limit(self, chat_id: int) -> Tuple[bool, float]:
        now = time.monotonic()
        last_ts = self.last_command_ts.get(chat_id, 0.0)
        elapsed = now - last_ts
        if elapsed < RATE_LIMIT_SECONDS:
            return False, RATE_LIMIT_SECONDS - elapsed
        self.last_command_ts[chat_id] = now
        return True, 0.0

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)


def validate_config() -> List[str]:
    errors: List[str] = []
    if not TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN no esta configurado.")
    if not VSCODE_BRIDGE_TOKEN:
        errors.append("VSCODE_BRIDGE_TOKEN no esta configurado.")
    if not VSCODE_BRIDGE_URL.startswith("http://127.0.0.1") and not VSCODE_BRIDGE_URL.startswith(
        "http://localhost"
    ):
        errors.append("VSCODE_BRIDGE_URL deberia apuntar a localhost por seguridad.")
    if not ALLOWED_CHAT_IDS and not AUTO_ALLOW_FIRST_CHAT and not Path(STATE_FILE_PATH).exists():
        errors.append(
            "Sin chats permitidos: completa ALLOWED_CHAT_IDS o activa AUTO_ALLOW_FIRST_CHAT."
        )
    return errors


def main() -> None:
    config_errors = validate_config()
    if config_errors:
        print("Configuracion invalida:", flush=True)
        for err in config_errors:
            print(f"- {err}", flush=True)
        raise SystemExit(1)

    bridge_client = BridgeClient(
        base_url=VSCODE_BRIDGE_URL,
        token=VSCODE_BRIDGE_TOKEN,
        timeout_sec=REQUEST_TIMEOUT_SEC,
    )

    bot = TelegramVsCodeBridgeBot(
        telegram_token=TELEGRAM_BOT_TOKEN,
        allowed_chat_ids=ALLOWED_CHAT_IDS,
        bridge_client=bridge_client,
    )
    bot.run_forever()


if __name__ == "__main__":
    main()
