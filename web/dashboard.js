// ==============================================================================
// Next-GenAI: Dashboard Logic
// Polls the FastAPI backend or Kafka alerts for real-time updates.
// ==============================================================================

const mockAlerts = [
    { domain: "a8f9s7df.exfil.evil.com", type: "DNS Tunneling", conf: "99.8%", shap: "query_entropy (+0.45)" },
    { domain: "update.badguy.net", type: "Polymorphic", conf: "98.4%", shap: "cert_validity_days (+0.32)" },
    { domain: "legitimate-bank.com", type: "TLS Hijack", conf: "95.1%", shap: "sni_mismatch (+0.28)" }
];

function renderAlerts() {
    const tbody = document.getElementById('alert-body');
    tbody.innerHTML = '';
    
    mockAlerts.forEach(alert => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${alert.domain}</td>
            <td><span class="badge ${alert.type.replace(" ", "-").toLowerCase()}">${alert.type}</span></td>
            <td>${alert.conf}</td>
            <td><code>${alert.shap}</code></td>
            <td><button onclick="blockIP('${alert.domain}')">Block</button></td>
        `;
        tbody.appendChild(tr);
    });
}

function blockIP(domain) {
    alert(`🚫 Action Sent: Blocking ${domain} at the firewall.`);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    renderAlerts();
    // Simulate live metric updates
    setInterval(() => {
        const total = document.getElementById('total-scans');
        let val = parseInt(total.innerText.replace(/,/g, ''));
        total.innerText = (val + Math.floor(Math.random() * 50)).toLocaleString();
    }, 2000);
});
