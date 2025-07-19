document.getElementById("predict-form").onsubmit = async function (e) {
  e.preventDefault();
  const features = document
    .getElementById("features")
    .value.split(",")
    .map(Number);
  const model = document.getElementById("model-select").value;
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features, model }),
  });
  const data = await res.json();
  document.getElementById("result").innerHTML = `<b>Prediction:</b> ${
    data.prediction === 1 ? "Fraudulent" : "Legitimate"
  }<br>
                 <b>Probability:</b> ${data.probability.toFixed(6)}<br>
         <b>Accuracy:</b> ${(data.accuracy * 100).toFixed(2)}%`;
};
