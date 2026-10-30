import unittest

from pydantic import ValidationError

from app.schemas import CustomerTicket, ModelAssessment


class SchemaTests(unittest.TestCase):
    def test_ticket_accepts_timezone_aware_timestamp(self):
        ticket = CustomerTicket.model_validate({
            "ticket_id": "T-1",
            "customer_id": "C-1",
            "raw_message": "Where is my order?",
            "timestamp": "2026-10-01T09:00:00Z",
        })
        self.assertEqual(ticket.ticket_id, "T-1")

    def test_ticket_rejects_missing_required_field_and_naive_timestamp(self):
        with self.assertRaises(ValidationError):
            CustomerTicket.model_validate({
                "ticket_id": "T-1", "customer_id": "C-1",
                "raw_message": "Help", "timestamp": "2026-10-01T09:00:00",
            })

    def test_assessment_rejects_out_of_range_confidence_and_unknown_fields(self):
        base = {
            "category": "shipping", "sentiment": "neutral", "confidence_score": 1.1,
        }
        with self.assertRaises(ValidationError):
            ModelAssessment.model_validate(base)
        with self.assertRaises(ValidationError):
            ModelAssessment.model_validate({
                "category": "shipping", "sentiment": "neutral", "confidence_score": 0.8,
                "unexpected": "not allowed",
            })


if __name__ == "__main__":
    unittest.main()
