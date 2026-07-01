import requests
import json

def test_chat(scenario_name, messages):
    payload = {"messages": messages}
    url = "http://127.0.0.1:8000/chat"
    
    print(f"\n{'='*60}")
    print(f"--- Running Test: {scenario_name} ---")
    print(f"{'='*60}")
    print("\nPayload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        print("\nResponse:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response text: {e.response.text}")

def main():
    # Test 1: Out of Scope Refusal
    test_1_messages = [
        {
            "role": "user",
            "content": "Can you give me some general advice on the legal risks of firing an employee in California?"
        }
    ]
    test_chat("Test 1: Out of Scope Refusal", test_1_messages)
    
    # Test 2: Grounded Comparison
    test_2_messages = [
        {
            "role": "user",
            "content": "What is the difference between the 'Occupational Personality Questionnaire OPQ32r' and the 'Global Skills Assessment'?"
        }
    ]
    test_chat("Test 2: Grounded Comparison", test_2_messages)
    
    # Test 3: Refine Behavior — user changes constraints mid-conversation
    test_3_messages = [
        {
            "role": "user",
            "content": "I need a personality assessment for hiring mid-level managers."
        },
        {
            "role": "assistant",
            "content": "For hiring mid-level managers, I recommend the Occupational Personality Questionnaire OPQ32r. It measures 32 workplace behaviour dimensions. Would you like me to finalize this recommendation?"
        },
        {
            "role": "user",
            "content": "Actually, I changed my mind. I need knowledge-based technical tests for software developers instead."
        }
    ]
    test_chat("Test 3: Refine Behavior (Constraint Change)", test_3_messages)

if __name__ == "__main__":
    main()
