from __future__ import annotations


def generate_recommendations(result: dict, inputs: dict, air_data: dict | None = None) -> list[str]:
    score = float(result["risk_score"])
    age = int(inputs["age"])
    exposure = float(inputs["exposure_min"])
    activity = str(inputs["activity"])
    asthma = int(inputs["asthma"])
    smoker = int(inputs["smoker"])
    mask_type = str(inputs["mask_type"])

    rec: list[str] = []

    if score < 20:
        rec.append("Air conditions are fairly manageable today. Regular outdoor activity is generally okay.")
    elif score < 40:
        rec.append("Air quality is moderate today. Work outside normally, but avoid overexertion if possible.")
    elif score < 65:
        rec.append("Air pollution is noticeable today. If you must stay outside, try to pace yourself and take small breaks.")
    else:
        rec.append("Air pollution is quite heavy today. Outdoor work may still be necessary, but protective habits matter more than usual.")

    rec.append("Drink water regularly. A few sips every 30 to 45 minutes is a good habit.")
    rec.append("Every hour, sit for 5 to 10 minutes and take slow deep breaths.")
    rec.append("If possible, wash your face and rinse your nose after long outdoor exposure.")

    if exposure >= 90:
        rec.append("Long outdoor exposure builds cumulative stress. Even short indoor breaks can help.")

    if activity in {"jog", "exercise"}:
        rec.append("Heavy activity increases pollution intake. Try to reduce pace during the most polluted periods.")

    if mask_type == "none":
        rec.append("A better mask, especially N95, can meaningfully reduce particle intake.")
    elif mask_type in {"cloth", "surgical"}:
        rec.append("Your mask offers some protection. A better-fitting mask would improve it further.")

    if asthma:
        rec.append("Because of asthma or wheezing risk, pay attention to coughing, chest tightness, or unusual breathlessness.")

    if smoker:
        rec.append("Smoking adds extra respiratory strain, so give yourself more recovery time today.")

    if 55 <= age < 70:
        rec.append("At your age, recovery may be slower than in younger adults, so short rest breaks are more important.")
    elif age >= 70:
        rec.append("Older lungs recover more slowly, so it is better to avoid continuous long exposure without rest.")

    if air_data is not None:
        pm25 = float(air_data.get("pm25", 0))
        if pm25 >= 60:
            rec.append("PM2.5 is relatively high. Avoid standing near traffic or smoke sources for long periods.")

    return rec
