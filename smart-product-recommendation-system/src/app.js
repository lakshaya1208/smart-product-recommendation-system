let model;
let data;

const $ = (id) => document.getElementById(id);

function sigmoid(value) {
  if (value >= 0) {
    const z = Math.exp(-value);
    return 1 / (1 + z);
  }
  const z = Math.exp(value);
  return z / (1 + z);
}

function predictProbability(features, category) {
  const numeric = features.map((value, index) =>
    (value - model.numeric_means[index]) / (model.numeric_scales[index] || 1)
  );
  const categoryVector = model.category_values.map((value) => value === category ? 1 : 0);
  const transformed = numeric.concat(categoryVector);
  const score = transformed.reduce(
    (sum, value, index) => sum + value * model.coefficients[index],
    model.intercept
  );
  return sigmoid(score);
}

function renderMetrics(metrics) {
  const values = [
    ["Accuracy", `${(metrics.accuracy * 100).toFixed(1)}%`],
    ["Precision", `${(metrics.precision * 100).toFixed(1)}%`],
    ["Recall", `${(metrics.recall * 100).toFixed(1)}%`],
    ["F1", metrics.f1.toFixed(3)],
    ["ROC-AUC", metrics.roc_auc.toFixed(3)]
  ];
  $("metrics").innerHTML = values.map(([label, value]) =>
    `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`
  ).join("");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  }[character]));
}

function getRecommendations(customerId) {
  const customer = data.customers.find((item) => item.customerId === customerId);
  if (!customer) throw new Error("Customer ID was not found.");

  const categoryPurchaseTotals = {};
  data.products.forEach((product) => {
    const interaction = data.interactions[`${customerId}:${product.productId}`];
    if (!interaction) return;
    categoryPurchaseTotals[product.category] =
      (categoryPurchaseTotals[product.category] || 0) + interaction.purchases;
  });

  return data.products
    .filter((product) => {
      const interaction = data.interactions[`${customerId}:${product.productId}`];
      return interaction && interaction.purchases === 0;
    })
    .map((product) => {
      const interaction = data.interactions[`${customerId}:${product.productId}`];
      const categoryHistory = (categoryPurchaseTotals[product.category] || 0) - interaction.purchases;
      const features = [
        product.price,
        interaction.views,
        interaction.cartAdditions,
        customer.previousPurchases,
        interaction.cartAdditions / Math.max(interaction.views, 1),
        categoryHistory
      ];
      return { ...product, probability: predictProbability(features, product.category) };
    })
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 5);
}

function renderRecommendations(recommendations) {
  $("recommendation-body").innerHTML = recommendations.map((item, index) => {
    const percentage = item.probability * 100;
    return `<tr>
      <td class="rank">0${index + 1}</td>
      <td><strong>${escapeHtml(item.productId)}</strong></td>
      <td>${escapeHtml(item.category)}</td>
      <td>₹${item.price.toLocaleString("en-IN", { maximumFractionDigits: 0 })}</td>
      <td><div class="probability-cell"><span>${percentage.toFixed(1)}%</span><div class="probability-track"><span style="width:${Math.max(percentage, 2)}%"></span></div></div></td>
    </tr>`;
  }).join("");
  $("result-count").textContent = `${recommendations.length} recommendations`;
  $("results").hidden = false;
  $("empty-state").hidden = true;
}

function setError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

async function loadProject() {
  try {
    const [modelResponse, dataResponse] = await Promise.all([
      fetch("models/model.json"),
      fetch("data/app_data.json")
    ]);
    if (!modelResponse.ok || !dataResponse.ok) throw new Error("Model or dataset artifact could not be loaded.");
    model = await modelResponse.json();
    data = await dataResponse.json();

    renderMetrics(model.metrics);
    $("interaction-count").textContent = `${data.interactions ? Object.keys(data.interactions).length.toLocaleString() : 0} interactions`;
    $("dataset-summary").textContent = `${data.customers.length} customers · ${data.products.length} products · reproducible seed 42`;

    const select = $("customer-id");
    select.innerHTML = data.customers.map((customer) => `<option value="${escapeHtml(customer.customerId)}">${escapeHtml(customer.customerId)}</option>`).join("");
    select.disabled = false;
    $("recommend-button").disabled = false;
    $("recommend-button").textContent = "Get Recommendations";
    $("model-status").innerHTML = "<span></span> Model loaded";
  } catch (error) {
    setError(error instanceof Error ? error.message : "Unable to load the project artifacts.");
    $("model-status").innerHTML = "<span></span> Model unavailable";
  }
}

$("recommendation-form").addEventListener("submit", (event) => {
  event.preventDefault();
  setError("");
  try {
    const recommendations = getRecommendations($("customer-id").value);
    if (recommendations.length < 5) throw new Error("Fewer than five unpurchased candidate products are available for this customer.");
    renderRecommendations(recommendations);
  } catch (error) {
    $("results").hidden = true;
    $("empty-state").hidden = false;
    setError(error instanceof Error ? error.message : "Unable to generate recommendations.");
  }
});

loadProject();
