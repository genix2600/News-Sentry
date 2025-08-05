def inject_chat_style():
    st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        padding: 1rem 2rem;
        max-width: 800px;
        margin: auto;
        font-family: 'Segoe UI', sans-serif;
        color: black;
    }
    .chat-bubble {
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        max-width: 100%;
        line-height: 1.6;
        background-color: #f4f4f4;
        word-wrap: break-word;
        color: black;
    }
    .user-msg {
        align-self: flex-end;
        background-color: #e0e0e0;
    }
    .bot-msg {
        align-self: flex-start;
        background-color: #ffffff;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

def run_chat_ui():
    st.set_page_config(page_title="Fake News Classifier", layout="centered")
    inject_chat_style()

    st.markdown("<h2 style='text-align: center; color: black;'>📰 Fake News Classifier Chat</h2>", unsafe_allow_html=True)
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    headline = st.text_input("💬 You:", placeholder="Enter a news headline here...", key="headline_input")

    if headline:
        # User message bubble
        st.markdown(f"<div class='chat-bubble user-msg'>{headline}</div>", unsafe_allow_html=True)

        processed = wordopt(headline)
        new_xv = vectorizer.transform([processed])

        predictions = []
        confidences = []

        for name, model in models.items():
            prediction = model.predict(new_xv)[0]
            proba = model.predict_proba(new_xv)[0]
            label = "Probably not fake news" if prediction == 1 else "Probably fake news"
            confidence = round(max(proba) * 100, 2)
            explanation = get_explanation(prediction)

            predictions.append(prediction)
            confidences.append((prediction, confidence))

            model_response = f"""
            **Model:** {name}  
            **Prediction:** {label}  
            **Confidence:** {confidence}%  
            {explanation}
            """
            st.markdown(f"<div class='chat-bubble bot-msg'>{model_response}</div>", unsafe_allow_html=True)

        final_vote = max(set(predictions), key=predictions.count)
        final_label = output_label(final_vote)
        real_count = predictions.count(1)
        fake_count = predictions.count(0)
        matching_conf = [conf for pred, conf in confidences if pred == final_vote]
        confidence_percent = round(sum(matching_conf) / len(matching_conf), 2) if matching_conf else 0.0

        final_response = f"""
        🧠 **Final Verdict**  
        Prediction: {final_label}  
        Votes → Real: {real_count}, Fake: {fake_count}  
        Average Confidence: {confidence_percent}%
        """
        st.markdown(f"<div class='chat-bubble bot-msg'>{final_response}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
