"""
Alert system with Feishu webhook integration.
Monitors system metrics and sends notifications when thresholds are exceeded.
"""

import requests
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger
from src.utils.config import get_section


class AlertSeverity:
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertRule:
    """Alert rule definition"""
    
    def __init__(
        self,
        name: str,
        condition: callable,
        severity: str = AlertSeverity.WARNING,
        duration: int = 300,  # seconds
        cooldown: int = 3600  # seconds
    ):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.duration = duration
        self.cooldown = cooldown
        self.triggered_at: Optional[datetime] = None
        self.last_alert_at: Optional[datetime] = None
    
    def should_alert(self, current_value: float) -> bool:
        """Check if alert should be triggered"""
        now = datetime.now()
        
        # Check condition
        if not self.condition(current_value):
            self.triggered_at = None
            return False
        
        # Check duration
        if self.triggered_at is None:
            self.triggered_at = now
            return False
        
        if (now - self.triggered_at).total_seconds() < self.duration:
            return False
        
        # Check cooldown
        if self.last_alert_at:
            if (now - self.last_alert_at).total_seconds() < self.cooldown:
                return False
        
        return True
    
    def mark_alerted(self):
        """Mark alert as sent"""
        self.last_alert_at = datetime.now()


class FeishuNotifier:
    """Feishu webhook notifier"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = AlertSeverity.WARNING,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send alert to Feishu"""
        if not self.webhook_url:
            logger.warning("Feishu webhook URL not configured")
            return False
        
        # Color based on severity
        color_map = {
            AlertSeverity.INFO: "blue",
            AlertSeverity.WARNING: "orange",
            AlertSeverity.CRITICAL: "red"
        }
        color = color_map.get(severity, "grey")
        
        # Build message card
        card = {
            "msg_type": "interactive",
            "card": {
                "config": {
                    "wide_screen_mode": True
                },
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"🚨 {title}"
                    },
                    "template": color
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": message
                        }
                    }
                ]
            }
        }
        
        # Add details if provided
        if details:
            detail_lines = []
            for key, value in details.items():
                detail_lines.append(f"**{key}**: {value}")
            
            card["card"]["elements"].append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "\n".join(detail_lines)
                }
            })
        
        # Add timestamp
        card["card"]["elements"].append({
            "tag": "div",
            "text": {
                "tag": "plain_text",
                "content": f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        })
        
        try:
            response = requests.post(
                self.webhook_url,
                json=card,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Alert sent to Feishu: {title}")
                return True
            else:
                logger.error(f"Failed to send Feishu alert: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending Feishu alert: {e}")
            return False


class AlertManager:
    """Alert manager"""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.notifiers: List[FeishuNotifier] = []
        self.enabled = False
        
        # Load configuration
        self.load_config()
    
    def load_config(self):
        """Load alert configuration"""
        try:
            config = get_section('alerting')
            self.enabled = config.get('enabled', False)
            
            if not self.enabled:
                logger.info("Alerting system disabled")
                return
            
            # Setup Feishu notifier
            feishu_config = config.get('feishu', {})
            webhook_url = feishu_config.get('webhook_url', '')
            
            if webhook_url:
                self.notifiers.append(FeishuNotifier(webhook_url))
                logger.info("Feishu notifier configured")
            
            # Register default rules
            self.register_default_rules(config.get('rules', []))
            
        except Exception as e:
            logger.error(f"Failed to load alert config: {e}")
    
    def register_default_rules(self, rule_configs: List[Dict]):
        """Register default alert rules"""
        for rule_config in rule_configs:
            name = rule_config.get('name', 'unknown')
            condition_str = rule_config.get('condition', '')
            duration = rule_config.get('duration', 300)
            severity = rule_config.get('severity', AlertSeverity.WARNING)
            
            # Parse condition (simple format: "metric > threshold")
            try:
                if '>' in condition_str:
                    metric, threshold = condition_str.split('>')
                    threshold = float(threshold.strip())
                    condition = lambda x: x > threshold
                elif '<' in condition_str:
                    metric, threshold = condition_str.split('<')
                    threshold = float(threshold.strip())
                    condition = lambda x: x < threshold
                else:
                    logger.warning(f"Invalid condition format: {condition_str}")
                    continue
                
                rule = AlertRule(
                    name=name,
                    condition=condition,
                    severity=severity,
                    duration=duration
                )
                self.rules.append(rule)
                logger.info(f"Registered alert rule: {name}")
                
            except Exception as e:
                logger.error(f"Failed to parse rule {name}: {e}")
    
    def check_metric(self, rule_name: str, value: float):
        """Check a metric against its rule"""
        if not self.enabled:
            return
        
        for rule in self.rules:
            if rule.name == rule_name:
                if rule.should_alert(value):
                    self.send_alert(
                        title=f"Alert: {rule.name}",
                        message=f"Metric {rule.name} triggered: {value}",
                        severity=rule.severity,
                        details={'value': value, 'rule': rule.name}
                    )
                    rule.mark_alerted()
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = AlertSeverity.WARNING,
        details: Optional[Dict[str, Any]] = None
    ):
        """Send alert through all notifiers"""
        for notifier in self.notifiers:
            notifier.send_alert(title, message, severity, details)


# Global alert manager instance
alert_manager = AlertManager()
