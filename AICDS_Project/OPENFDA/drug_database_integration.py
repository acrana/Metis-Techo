import aiohttp
from typing import Dict, Optional

class DrugDatabase:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.fda.gov/drug/label.json"

    async def get_drug_info(self, drug_name: str) -> Optional[Dict]:
        """Get drug information from OpenFDA"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "api_key": self.api_key,
                    "search": f"openfda.generic_name:{drug_name}",
                    "limit": 1
                }
                
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "results" in data and len(data["results"]) > 0:
                            result = data["results"][0]
                            return {
                                "interactions": result.get("drug_interactions", []),
                                "warnings": result.get("warnings_and_cautions", []),
                                "monitoring": result.get("boxed_warning", []),
                                "indications": result.get("indications_and_usage", []),
                                "dosage": result.get("dosage_and_administration", [])
                            }
            return None
        except Exception as e:
            print(f"Error getting drug info: {str(e)}")
            return None