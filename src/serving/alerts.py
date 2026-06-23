import os
import requests
import yaml
import logging

# Configuração simples de logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def dispatch_alert(alert: dict) -> None:
    """
    Envia alertas climáticos para múltiplos canais (Console, Telegram e Webhook).
    """
    config = load_config()
    
    # 1. Formatando a mensagem
    level_emoji = {
        'NORMAL': '🟢',
        'ATENÇÃO': '🟡',
        'ALERTA': '🟠',
        'EMERGÊNCIA': '🔴'
    }.get(alert['level'], '⚪')
    
    timestamp_str = alert['timestamp']
    if hasattr(timestamp_str, 'strftime'):
        timestamp_str = timestamp_str.strftime('%d/%m/%Y %H:%M')
        
    message = (
        f"{level_emoji} [HexClima] [{alert['level']}] Anomalia climática detectada no RS\n"
        f"Horário: {timestamp_str} UTC\n"
        f"Estação: {alert.get('season', 'N/A').capitalize()}\n"
        f"Score de Reconstrução: {alert['reconstruction_error']:.6f}\n"
        f"Threshold Esperado: {alert['threshold']:.6f}\n"
        f"Severidade: {alert['severity']*100:.2f}% acima do limiar\n"
        f"Ação Recomendada: Monitorar e seguir protocolos da Defesa Civil RS."
    )
    
    # Sempre imprimir no console/log
    logger.warning("\n================= ALERTA DISPARADO =================")
    logger.warning(message)
    logger.warning("====================================================\n")
    
    # 2. Despachar para Telegram (Se configurado)
    telegram_token = config['serving'].get('telegram_token')
    chat_id = config['serving'].get('chat_id')
    
    if telegram_token and telegram_token != "YOUR_TELEGRAM_TOKEN":
        telegram_url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        try:
            res = requests.post(telegram_url, json={'chat_id': chat_id, 'text': message}, timeout=5)
            if res.status_code == 200:
                logger.info("Notificação enviada ao Telegram com sucesso.")
            else:
                logger.error(f"Falha ao enviar ao Telegram: {res.status_code} - {res.text}")
        except Exception as e:
            logger.error(f"Erro de conexão com API do Telegram: {e}")
    else:
        logger.info("Envio de Telegram ignorado (Token/Chat ID padrão ou não configurado).")
        
    # 3. Despachar para Webhook da Defesa Civil / SEMA-RS
    webhook_url = config['serving'].get('defesa_civil_webhook')
    if webhook_url:
        # Converter timestamp para string serializável
        payload = alert.copy()
        if hasattr(payload['timestamp'], 'isoformat'):
            payload['timestamp'] = payload['timestamp'].isoformat()
            
        try:
            res = requests.post(webhook_url, json=payload, timeout=5)
            if res.status_code in [200, 201]:
                logger.info("Alerta despachado via Webhook com sucesso.")
            else:
                # Como é uma simulação, pode falhar silenciosamente se o endpoint não estiver ativo
                logger.info(f"Retorno do Webhook (simulado/pendente): {res.status_code} (URL: {webhook_url})")
        except Exception as e:
            logger.info(f"Simulando chamada Webhook: {e} (Webhook offline ou fictício).")

if __name__ == "__main__":
    # Teste rápido de dispatch
    from datetime import datetime
    test_alert = {
        'timestamp': datetime.utcnow(),
        'reconstruction_error': 0.1245,
        'threshold': 0.0800,
        'severity': 0.5562,
        'level': 'ALERTA',
        'is_anomaly': True,
        'season': 'primavera'
    }
    dispatch_alert(test_alert)
