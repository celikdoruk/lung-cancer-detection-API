import requests

def cancer_outcome(data: dict, url_path: str) -> dict:
    response = requests.post(url=url_path, json=data)
    data = response.json()
    return data

def main():
    data_to_pass = {
    "age": 35, "gender": "male", "pack_years": 65, "radon_exposure": "high", "asbestos_exposure": "yes",
    "secondhand_smoke_exposure": "yes", "copd_diagnosis": "no","alcohol_consumption": "heavy","family_history": "yes"}
    
    url_path = "http://127.0.0.1:8000/"

    try:
        response = cancer_outcome(data=data_to_pass, url_path=url_path)
        print(f"Api Response: {response["model_prediction"]}")
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Request error: {e}")
        

if __name__ == "__main__":
    main()
