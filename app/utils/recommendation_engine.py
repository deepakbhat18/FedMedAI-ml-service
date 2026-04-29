import os
import json
import requests
from typing import Dict, List, Optional


class RecommendationEngine:
    """
    AI-powered medical recommendation engine.
    Uses OpenRouter API with free DeepSeek/Llama models.
    Falls back to rule-based if API unavailable.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "meta-llama/llama-3.1-8b-instruct"  # FREE model
        self.enabled = bool(self.api_key and self.api_key != "your-api-key-here")

        if self.enabled:
            print(f"  ✅ AI Recommendations: OpenRouter enabled ({self.model})")
        else:
            print(f"  ⚠️ AI Recommendations: Using rule-based fallback")

    def generate(
        self,
        patient_features: Dict,
        predictions: List[Dict],
        patient_info: Optional[Dict] = None
    ) -> Dict:
        """
        Generate personalized medical recommendations.

        Args:
            patient_features: Patient vitals and lab values
            predictions: ML model predictions (list of disease results)
            patient_info: Optional patient metadata (age, gender etc)

        Returns:
            Recommendations dictionary
        """
        # Only recommend for HIGH and MODERATE risk
        relevant = [
            p for p in predictions
            if p.get("risk_level") in ["HIGH", "MODERATE"]
        ]

        if not relevant:
            return self._healthy_response()

        if self.enabled:
            try:
                return self._call_openrouter(patient_features, predictions)
            except Exception as e:
                print(f"  ⚠️ OpenRouter failed: {e}. Using fallback.")
                return self._rule_based(predictions, patient_features)
        else:
            return self._rule_based(predictions, patient_features)

    def _call_openrouter(self, features: Dict, predictions: List[Dict]) -> Dict:
        """Call OpenRouter API for AI recommendations."""

        prompt = self._build_prompt(features, predictions)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "FL Medical Diagnosis System"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert medical AI assistant. "
                        "Analyze patient data and AI predictions to provide "
                        "personalized, actionable medical recommendations. "
                        "Always respond in valid JSON format only. "
                        "Be specific, concise, and medically accurate."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2500,
            "temperature": 0.2,
        }

        print(f"  📡 Calling OpenRouter ({self.model})...")
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(
                f"API error {response.status_code}: {response.text[:200]}"
            )

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"  ✅ OpenRouter response received")

        return self._parse_ai_response(content, predictions)

    def _build_prompt(self, features: Dict, predictions: List[Dict]) -> str:
        """Build comprehensive prompt for the AI model."""

        # Format feature values with normal ranges
        feature_analysis = []

        bp = features.get("bloodPressure", features.get("blood_pressure", 0))
        if bp:
            status = "⚠️ HIGH" if float(bp) > 140 else "✅ Normal"
            feature_analysis.append(f"Blood Pressure: {bp} mmHg ({status}, normal: <120)")

        glucose = features.get("glucose", 0)
        if glucose:
            status = "⚠️ HIGH" if float(glucose) > 126 else "✅ Normal"
            feature_analysis.append(f"Blood Glucose: {glucose} mg/dl ({status}, normal: 70-100)")

        bmi = features.get("bmi", 0)
        if bmi:
            bmi_val = float(bmi)
            if bmi_val > 30:
                status = "⚠️ Obese"
            elif bmi_val > 25:
                status = "⚠️ Overweight"
            else:
                status = "✅ Normal"
            feature_analysis.append(f"BMI: {bmi} ({status}, normal: 18.5-24.9)")

        cholesterol = features.get("cholesterol", 0)
        if cholesterol:
            status = "⚠️ HIGH" if float(cholesterol) > 200 else "✅ Normal"
            feature_analysis.append(f"Cholesterol: {cholesterol} mg/dl ({status}, normal: <200)")

        creatinine = features.get("creatinine", 0)
        if creatinine:
            status = "⚠️ HIGH" if float(creatinine) > 1.2 else "✅ Normal"
            feature_analysis.append(f"Creatinine: {creatinine} mg/dl ({status}, normal: 0.6-1.2)")

        bilirubin = features.get("bilirubin", 0)
        if bilirubin:
            status = "⚠️ HIGH" if float(bilirubin) > 1.2 else "✅ Normal"
            feature_analysis.append(f"Bilirubin: {bilirubin} mg/dl ({status}, normal: 0.1-1.2)")

        hemoglobin = features.get("hemoglobin", 0)
        if hemoglobin:
            hgb = float(hemoglobin)
            gender = features.get("gender", 0)
            normal_low = 13.5 if gender == 1 else 12.0
            status = "⚠️ LOW (Anemia)" if hgb < normal_low else "✅ Normal"
            feature_analysis.append(f"Hemoglobin: {hemoglobin} g/dl ({status})")

        alt = features.get("altEnzyme", features.get("alt_enzyme", 0))
        if alt:
            status = "⚠️ HIGH" if float(alt) > 40 else "✅ Normal"
            feature_analysis.append(f"ALT Enzyme: {alt} U/L ({status}, normal: <40)")

        smoking = features.get("smoking", 0)
        feature_analysis.append(f"Smoking: {'⚠️ YES' if float(smoking) == 1 else '✅ No'}")

        diabetes_history = features.get("diabetesHistory", features.get("diabetes_history", 0))
        feature_analysis.append(
            f"Diabetes History: {'⚠️ YES' if float(diabetes_history) == 1 else '✅ No'}")

        family_history = features.get("familyHistory", features.get("family_history", 0))
        feature_analysis.append(
            f"Family History of Disease: {'⚠️ YES' if float(family_history) == 1 else '✅ No'}")

        age = features.get("age", "N/A")
        gender_str = "Male" if float(features.get("gender", 0)) == 1 else "Female"
        feature_analysis.append(f"Age: {age} years")
        feature_analysis.append(f"Gender: {gender_str}")

        # Format predictions
        pred_lines = []
        for pred in sorted(predictions, key=lambda x: x.get("confidence", 0), reverse=True):
            conf = pred.get("confidence", 0) * 100
            pred_lines.append(
                f"- {pred.get('disease_name', pred.get('disease_type'))}: "
                f"{conf:.1f}% → {pred.get('risk_level', 'UNKNOWN')}"
            )

        prompt = f"""Analyze this patient's medical data and AI predictions.

PATIENT VITALS & LAB VALUES:
{chr(10).join(feature_analysis)}

AI DISEASE PREDICTIONS:
{chr(10).join(pred_lines)}

Provide personalized medical recommendations in this EXACT JSON format:

{{
    "disease_recommendations": [
        {{
            "disease_type": "heart_disease",
            "disease_name": "Heart Disease",
            "risk_level": "HIGH",
            "urgency": "IMMEDIATE",
            "urgency_label": "See doctor today",
            "explanation": "Specific explanation based on patient values",
            "key_risk_factors": [
                "Specific factor with patient value"
            ],
            "tests_recommended": [
                "Specific test name"
            ],
            "specialists_to_consult": [
                "Specific specialist"
            ],
            "lifestyle_changes": [
                "Specific actionable change"
            ],
            "medications_note": "Brief medication note for doctor"
        }}
    ],
    "overall_summary": "2-3 sentence patient health summary",
    "most_urgent": "Most critical action needed right now",
    "follow_up_timeline": "When to schedule follow-up",
    "positive_notes": "Any positive health aspects",
    "general_advice": [
        "General health tip 1",
        "General health tip 2"
    ]
}}

RULES:
1. Only include HIGH and MODERATE risk diseases in disease_recommendations
2. Be specific to patient values (mention actual numbers)
3. urgency must be one of: IMMEDIATE, WITHIN_48_HOURS, WITHIN_1_WEEK, WITHIN_1_MONTH, ROUTINE
4. Return ONLY valid JSON, no markdown, no extra text"""

        return prompt

    def _parse_ai_response(self, content: str, predictions: List[Dict]) -> Dict:
        """Parse AI response and clean up."""
        try:
            # Remove markdown code blocks if present
            clean = content.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()

            # Find JSON object
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start != -1 and end > start:
                clean = clean[start:end]

            data = json.loads(clean)
            data["source"] = "OpenRouter AI"
            data["model_used"] = self.model
            data["ai_powered"] = True
            return data

        except Exception as e:
            print(f"  ⚠️ JSON parse error: {e}")
            print(f"  Raw response: {content[:300]}")
            return self._rule_based(predictions, {})

    def _rule_based(self, predictions: List[Dict], features: Dict) -> Dict:
        """
        Rule-based fallback recommendations.
        Used when OpenRouter is unavailable.
        """
        RULES = {
            "heart_disease": {
                "urgency": "WITHIN_1_WEEK",
                "urgency_label": "See cardiologist within 1 week",
                "tests": [
                    "ECG (Electrocardiogram)",
                    "Echocardiogram",
                    "Lipid profile (cholesterol test)",
                    "Troponin I/T (heart damage markers)",
                    "Stress test (Treadmill test)"
                ],
                "specialists": ["Cardiologist", "General Physician"],
                "lifestyle": [
                    "Heart-healthy diet (low saturated fat)",
                    "30 minutes moderate exercise daily",
                    "Stop smoking immediately",
                    "Reduce salt intake to <5g/day",
                    "Limit alcohol consumption",
                    "Manage stress with relaxation techniques",
                    "Monitor blood pressure daily"
                ],
                "medications_note": "Doctor may consider statins, beta-blockers, or ACE inhibitors"
            },
            "diabetes": {
                "urgency": "WITHIN_1_WEEK",
                "urgency_label": "Blood test within 1 week",
                "tests": [
                    "HbA1c test (3-month glucose average)",
                    "Fasting blood glucose test",
                    "Oral glucose tolerance test (OGTT)",
                    "Kidney function test (diabetes affects kidneys)",
                    "Urine microalbumin test",
                    "Lipid profile"
                ],
                "specialists": ["Endocrinologist", "Dietitian/Nutritionist"],
                "lifestyle": [
                    "Low-carbohydrate, low-sugar diet",
                    "Eliminate sugary drinks and processed foods",
                    "Monitor blood glucose daily (morning fasting)",
                    "30 minutes walking after each meal",
                    "Target 5-10% weight reduction if overweight",
                    "Regular eye check (diabetes affects vision)",
                    "Foot care examination regularly"
                ],
                "medications_note": "Doctor may consider Metformin as first-line treatment"
            },
            "kidney_disease": {
                "urgency": "WITHIN_1_WEEK",
                "urgency_label": "Kidney function tests within 1 week",
                "tests": [
                    "Serum creatinine and BUN",
                    "eGFR (estimated Glomerular Filtration Rate)",
                    "Complete urine analysis",
                    "Urine protein-to-creatinine ratio",
                    "Kidney ultrasound",
                    "Electrolytes (sodium, potassium)"
                ],
                "specialists": ["Nephrologist (kidney specialist)"],
                "lifestyle": [
                    "Reduce protein intake (limit red meat)",
                    "Low sodium diet (<2g/day)",
                    "Stay well hydrated (6-8 glasses water)",
                    "Avoid NSAIDs (ibuprofen, naproxen)",
                    "Control blood pressure strictly (<130/80)",
                    "Avoid contrast dyes in imaging if possible",
                    "Stop smoking (worsens kidney disease)"
                ],
                "medications_note": "Doctor may adjust existing medications for kidney safety"
            },
            "liver_disease": {
                "urgency": "WITHIN_1_MONTH",
                "urgency_label": "Liver function tests within 1 month",
                "tests": [
                    "Complete Liver Function Test (LFT)",
                    "Hepatitis B and C screening",
                    "Liver ultrasound",
                    "FibroScan (liver stiffness)",
                    "Biopsy if advanced disease suspected",
                    "AFP (alpha-fetoprotein) for cancer screening"
                ],
                "specialists": ["Hepatologist", "Gastroenterologist"],
                "lifestyle": [
                    "Stop alcohol consumption completely",
                    "Low-fat, high-fiber diet",
                    "Avoid unnecessary medications and supplements",
                    "Maintain healthy weight",
                    "Avoid raw/undercooked shellfish",
                    "Get vaccinated for Hepatitis A and B",
                    "Regular moderate exercise"
                ],
                "medications_note": "Avoid acetaminophen (Tylenol) - harmful to damaged liver"
            },
            "stroke": {
                "urgency": "IMMEDIATE",
                "urgency_label": "URGENT - See neurologist immediately",
                "tests": [
                    "Brain MRI or CT scan",
                    "Carotid ultrasound (neck arteries)",
                    "ECG (detect atrial fibrillation)",
                    "24-hour Holter monitor",
                    "Blood coagulation tests",
                    "Complete blood count (CBC)"
                ],
                "specialists": [
                    "Neurologist (URGENT)",
                    "Cardiologist",
                    "Vascular surgeon if needed"
                ],
                "lifestyle": [
                    "Stop smoking IMMEDIATELY (2x stroke risk)",
                    "Monitor blood pressure twice daily",
                    "Strict salt restriction (<5g/day)",
                    "Mediterranean diet (heart and brain healthy)",
                    "Limit alcohol to <1 drink/day",
                    "Regular moderate aerobic exercise",
                    "Stress management (meditation, yoga)",
                    "Learn stroke warning signs (FAST: Face, Arms, Speech, Time)"
                ],
                "medications_note": "Doctor may prescribe blood thinners or antiplatelets"
            }
        }

        recommendations = []
        for pred in sorted(predictions, key=lambda x: x.get("confidence", 0), reverse=True):
            if pred.get("risk_level") not in ["HIGH", "MODERATE"]:
                continue

            disease_key = pred.get("disease_type", "")
            rules = RULES.get(disease_key, {})
            confidence = pred.get("confidence", 0) * 100

            # Build feature-specific explanation
            explanation_parts = []
            if disease_key == "diabetes":
                glucose = features.get("glucose", 0)
                bmi = features.get("bmi", 0)
                if glucose and float(glucose) > 126:
                    explanation_parts.append(
                        f"Blood glucose ({glucose} mg/dl) significantly above normal (70-100 mg/dl)")
                if bmi and float(bmi) > 30:
                    explanation_parts.append(f"BMI ({bmi}) indicates obesity - major diabetes risk factor")
                if features.get("diabetesHistory", 0) == 1:
                    explanation_parts.append("Positive diabetes family history detected")

            elif disease_key == "heart_disease":
                bp = features.get("bloodPressure", 0)
                chol = features.get("cholesterol", 0)
                if bp and float(bp) > 140:
                    explanation_parts.append(f"Blood pressure ({bp} mmHg) is elevated (normal: <120)")
                if chol and float(chol) > 200:
                    explanation_parts.append(f"Cholesterol ({chol} mg/dl) is high (normal: <200)")
                if features.get("smoking", 0) == 1:
                    explanation_parts.append("Active smoking is a major heart disease risk factor")

            elif disease_key == "stroke":
                age = features.get("age", 0)
                bp = features.get("bloodPressure", 0)
                if age and float(age) > 55:
                    explanation_parts.append(f"Age ({age}) significantly increases stroke risk")
                if bp and float(bp) > 140:
                    explanation_parts.append(f"Hypertension (BP: {bp}) is leading cause of stroke")
                if features.get("smoking", 0) == 1:
                    explanation_parts.append("Smoking doubles the risk of stroke")

            elif disease_key == "kidney_disease":
                creatinine = features.get("creatinine", 0)
                bp = features.get("bloodPressure", 0)
                if creatinine and float(creatinine) > 1.2:
                    explanation_parts.append(
                        f"Elevated creatinine ({creatinine} mg/dl) indicates reduced kidney function")
                if bp and float(bp) > 140:
                    explanation_parts.append("Hypertension is a leading cause of kidney disease")

            elif disease_key == "liver_disease":
                bilirubin = features.get("bilirubin", 0)
                alt = features.get("altEnzyme", features.get("alt_enzyme", 0))
                if bilirubin and float(bilirubin) > 1.2:
                    explanation_parts.append(
                        f"Elevated bilirubin ({bilirubin} mg/dl) indicates liver dysfunction")
                if alt and float(alt) > 40:
                    explanation_parts.append(
                        f"High ALT enzyme ({alt} U/L) indicates liver cell damage")

            explanation = ". ".join(explanation_parts) if explanation_parts else (
                f"AI model detected {pred.get('risk_level')} risk with {confidence:.1f}% confidence"
            )

            recommendations.append({
                "disease_type": disease_key,
                "disease_name": pred.get("disease_name", disease_key),
                "risk_level": pred.get("risk_level"),
                "urgency": rules.get("urgency", "WITHIN_1_MONTH"),
                "urgency_label": rules.get("urgency_label", "See doctor soon"),
                "explanation": explanation,
                "key_risk_factors": explanation_parts if explanation_parts else ["Multiple risk factors detected"],
                "tests_recommended": rules.get("tests", []),
                "specialists_to_consult": rules.get("specialists", ["General Physician"]),
                "lifestyle_changes": rules.get("lifestyle", []),
                "medications_note": rules.get("medications_note", "Consult doctor for medication advice")
            })

        # Overall summary
        high_risk_count = len([r for r in recommendations if r["risk_level"] == "HIGH"])
        moderate_risk_count = len([r for r in recommendations if r["risk_level"] == "MODERATE"])

        if high_risk_count > 0:
            summary = (
                f"Patient shows HIGH risk for {high_risk_count} disease(s) and "
                f"MODERATE risk for {moderate_risk_count} disease(s). "
                f"Immediate medical consultation is strongly recommended."
            )
            most_urgent = recommendations[0]["urgency_label"] if recommendations else "See doctor immediately"
        else:
            summary = (
                f"Patient shows MODERATE risk for {moderate_risk_count} disease(s). "
                f"Schedule medical consultation within 1-2 weeks."
            )
            most_urgent = "Schedule appointment within 1-2 weeks"

        return {
            "disease_recommendations": recommendations,
            "overall_summary": summary,
            "most_urgent": most_urgent,
            "follow_up_timeline": "2 weeks after initial tests",
            "positive_notes": "Early detection gives better treatment outcomes",
            "general_advice": [
                "Stay hydrated - drink 8 glasses of water daily",
                "Get 7-8 hours of quality sleep every night",
                "Practice stress management (meditation, deep breathing)",
                "Schedule regular health checkups every 6 months",
                "Maintain a health journal to track symptoms"
            ],
            "source": "Rule-based (OpenRouter unavailable)",
            "ai_powered": False
        }

    def _healthy_response(self) -> Dict:
        """Response when all risks are minimal."""
        return {
            "disease_recommendations": [],
            "overall_summary": "Great news! Patient shows minimal risk for all 5 checked diseases. Continue maintaining healthy lifestyle.",
            "most_urgent": "Annual routine checkup",
            "follow_up_timeline": "Annual checkup in 12 months",
            "positive_notes": "All disease risk levels are within acceptable range.",
            "general_advice": [
                "Maintain balanced diet with fruits and vegetables",
                "Exercise 150 minutes per week (moderate intensity)",
                "Get annual health checkup",
                "Avoid smoking and excessive alcohol",
                "Maintain healthy BMI (18.5-24.9)",
                "Stay socially active and mentally engaged",
                "Get adequate sleep (7-8 hours)"
            ],
            "source": "Rule-based",
            "ai_powered": False
        }


# Global instance
recommendation_engine = RecommendationEngine()