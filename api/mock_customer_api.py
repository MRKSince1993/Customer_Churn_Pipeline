# api/mock_customer_api.py

from flask import Flask, jsonify, request
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# Generate some stable fake data for consistency
random.seed(42)
CUSTOMER_INTERACTIONS = {
    f"CUST_{i}": {
        "support_calls": random.randint(0, 5),
        "satisfaction_score": random.randint(1, 5),
        "last_interaction_days_ago": random.randint(1, 90)
    } for i in range(10000)
}

@app.route('/customer_interactions', methods=)
def get_customer_interactions():
    """
    API endpoint to fetch interaction data for a list of customer IDs.
    """
    customer_ids = request.json.get('customer_ids')
    if not customer_ids:
        return jsonify({"error": "customer_ids not provided"}), 400
    
    response_data = {}
    for cid in customer_ids:
        # Return pre-generated data or a default if not found
        response_data[cid] = CUSTOMER_INTERACTIONS.get(cid, {
            "support_calls": 0,
            "satisfaction_score": 3,
            "last_interaction_days_ago": 91
        })
        
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)