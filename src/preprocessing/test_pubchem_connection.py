"""
============================================================
 PubChem Connection Tester
 Run this FIRST to diagnose why SMILES fetching is failing.
============================================================
"""

import requests

def test_connection():
    print("="*60)
    print("  TESTING PUBCHEM API CONNECTION")
    print("="*60)

    test_url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        "Erlotinib/property/CanonicalSMILES,IsomericSMILES/JSON"
    )

    print(f"\nTesting URL:\n  {test_url}\n")

    try:
        response = requests.get(test_url, timeout=15)
        print(f"Status Code : {response.status_code}")
        print(f"Headers     : {dict(response.headers)}")
        print(f"\nResponse Body (first 1000 chars):")
        print(response.text[:1000])

        if response.status_code == 200:
            print("\n[SUCCESS] PubChem API is reachable and working!")
            data = response.json()
            smiles = data['PropertyTable']['Properties'][0].get('IsomericSMILES')
            print(f"           Erlotinib SMILES: {smiles}")
        elif response.status_code == 403:
            print("\n[BLOCKED] 403 Forbidden -- likely a firewall, proxy,")
            print("          antivirus, or corporate network blocking the request.")
        elif response.status_code == 404:
            print("\n[NOT FOUND] 404 -- PubChem doesn't recognize 'Erlotinib' by that")
            print("            exact name (unlikely, this is a very common drug).")
        else:
            print(f"\n[UNEXPECTED] Got status code {response.status_code}")

    except requests.exceptions.SSLError as e:
        print(f"\n[SSL ERROR] {e}")
        print("  -> Likely a corporate proxy doing SSL inspection,")
        print("     or an outdated certifi package.")
    except requests.exceptions.ConnectionError as e:
        print(f"\n[CONNECTION ERROR] {e}")
        print("  -> No internet reaching pubchem.ncbi.nlm.nih.gov.")
        print("     Check Wi-Fi, VPN, firewall, or institutional network blocks.")
    except requests.exceptions.Timeout as e:
        print(f"\n[TIMEOUT] {e}")
        print("  -> Request took too long. Network may be slow or blocked.")
    except Exception as e:
        print(f"\n[UNKNOWN ERROR] {type(e).__name__}: {e}")

    print("\n" + "="*60)


if __name__ == '__main__':
    test_connection()
