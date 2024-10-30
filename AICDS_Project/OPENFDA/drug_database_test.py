import asyncio
from datetime import datetime
from pprint import pprint
from drug_database_integration import DrugDatabase

async def main():
    # Initialize the drug database with your API key
    api_key = "PE0obWbVvRlH38Lbdx8uhuDk9vCNhHlmqSHGq6d2"
    db = DrugDatabase(api_key)
    
    # List of drugs to test
    drugs = ["warfarin", "metformin", "amiodarone"]
    
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    for drug in drugs:
        print(f"\nTesting drug: {drug.upper()}")
        print("-" * 30)
        
        result = await db.get_drug_info(drug)
        if result:
            print("\nFound information:")
            if result.get("indications"):
                print("\nINDICATIONS:")
                print(result["indications"][0][:200] + "...")
            
            if result.get("warnings"):
                print("\nWARNINGS:")
                print(result["warnings"][0][:200] + "...")
            
            if result.get("interactions"):
                print("\nINTERACTIONS:")
                print(result["interactions"][0][:200] + "...")
        else:
            print("No information found")
            
        print("-" * 30)
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())
