import json
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any
from loguru import logger

class AlphaZooPersistence:
    def __init__(self, storage_dir: str = "data/alpha_zoo"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
    def save_node(self, node: Any, metadata: Dict[str, Any] = None):
        """
        Save an AlphaNode to a JSON file.
        """
        formula = node.formula
        # Generate a unique ID based on the formula
        formula_id = hashlib.md5(formula.encode()).hexdigest()[:12]
        
        data = {
            "id": formula_id,
            "formula": formula,
            "metrics": node.metrics,
            "timestamp": datetime.now().isoformat(),
            "name": getattr(node, 'name', 'unknown'),
            "description": getattr(node, 'description', ''),
            "metadata": metadata or {}
        }
        
        file_path = os.path.join(self.storage_dir, f"alpha_{formula_id}.json")
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.debug(f"Saved alpha factor {formula_id} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save alpha {formula_id}: {e}")

    def save_zoo(self, zoo: List[Any], task_name: str = "default"):
        """
        Save the entire Alpha Zoo.
        """
        logger.info(f"Saving {len(zoo)} factors to Zoo...")
        for node in zoo:
            self.save_node(node, {"task": task_name})

    def load_all(self) -> List[Dict[str, Any]]:
        """
        Load all discovered alphas from the storage directory.
        """
        alphas = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json") and filename.startswith("alpha_"):
                path = os.path.join(self.storage_dir, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        alphas.append(json.load(f))
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
        
        # Sort by RankIC by default
        alphas.sort(key=lambda x: abs(x['metrics'].get('rank_ic', 0)), reverse=True)
        return alphas

    def export_to_csv(self, output_path: str = "data/alpha_zoo_summary.csv"):
        """
        Export factor summary to CSV for easy spreadsheet viewing.
        """
        import pandas as pd
        alphas = self.load_all()
        if not alphas:
            return
            
        flat_data = []
        for a in alphas:
            row = {
                "id": a['id'],
                "formula": a['formula'],
                "rank_ic": a['metrics'].get('rank_ic'),
                "ic_ir": a['metrics'].get('ic_ir'),
                "fitness": a['metrics'].get('fitness'),
                "timestamp": a['timestamp']
            }
            flat_data.append(row)
            
        df = pd.DataFrame(flat_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported zoo summary to {output_path}")
