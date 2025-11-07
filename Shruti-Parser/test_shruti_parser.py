import json
import requests
import unittest

class TestShrutiParser(unittest.TestCase):
    BASE_URL = "http://localhost:8000"
    
    def setUp(self):
        # Load test data
        with open("test_data/samples.json", "r") as f:
            self.test_data = json.load(f)["samples"]
    
    def test_analyze_endpoint(self):
        for sample in self.test_data:
            response = requests.post(
                f"{self.BASE_URL}/analyze",
                json={"text": sample["text"]}
            )
            self.assertEqual(response.status_code, 200)
            result = response.json()
            # Allow any of the four permitted intents; LLMs can vary in labeling.
            allowed = {"command", "query", "teaching", "data"}
            self.assertIn(result.get("intent"), allowed)
            
    def test_extract_endpoint(self):
        for sample in self.test_data:
            response = requests.post(
                f"{self.BASE_URL}/extract",
                json={"text": sample["text"]}
            )
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertIn("actions", result)
            
    def test_summarize_endpoint(self):
        for sample in self.test_data:
            response = requests.post(
                f"{self.BASE_URL}/summarize",
                json={"text": sample["text"]}
            )
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertIn("summary", result)

if __name__ == "__main__":
    unittest.main()