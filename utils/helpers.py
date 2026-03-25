from services.bse_client import BSEClient

def get_bse_client(config):
    """Helper function to get BSE client instance"""
    return BSEClient()