"""
Example 1: Enterprise Support Ticket Router.
Demonstrates dynamic Choice decision using JevClient with rich semantic option descriptions.
"""
import sys
import os

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.jev import JevClient
from client.schemas import OptionItem


def run_ticket_router():
    print("=== JEVELOPER: Real-World Customer Support Router ===")
    client = JevClient()

    # Incoming customer ticket
    ticket_text = (
        "Hello, I was checking my monthly statement and noticed a $129.99 charge from a merchant "
        "called 'CloudCompute Dublin' that I never signed up for or authorized. Please reverse this immediately."
    )

    # Dynamic candidate routing queues with rich semantic descriptions
    routing_options = [
        OptionItem(
            id="fraud_and_disputes",
            description="Unauthorized transactions, fraudulent charges, and refund chargebacks."
        ),
        OptionItem(
            id="card_management",
            description="Lost cards, stolen cards, pin reset, and card activation."
        ),
        OptionItem(
            id="technical_support",
            description="Mobile app bugs, login errors, password resets, and API issues."
        ),
        OptionItem(
            id="general_inquiry",
            description="Branch hours, interest rates, rewards points, and terms of service."
        )
    ]

    print(f"\n[Ticket State]:\n\"{ticket_text}\"")
    print("\n[Executing Jev Decision...]")

    res = client.choice(
        state=ticket_text,
        question="Which department should handle this customer issue?",
        options=routing_options
    )

    print("\n--- Decision Result ---")
    print(f"Selected Queue : {res.selected}")
    print(f"Confidence     : {res.confidence * 100:.2f}%")
    print(f"Latency        : {res.latency_ms:.2f} ms")
    print("\nProbability Distribution:")
    for opt_id, prob in res.probabilities.items():
        bar = "█" * int(prob * 30)
        print(f"  {opt_id:<20} {prob * 100:>5.1f}% | {bar}")


if __name__ == "__main__":
    run_ticket_router()
