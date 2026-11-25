# Next-GenAI: Next-Generation AI for Advanced Threat Detection and Security Enhancement in DNS over HTTPS

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-red.svg)](https://pytorch.org/)
[![Apache Flink](https://img.shields.io/badge/Apache%20Flink-1.18-orange.svg)](https://flink.apache.org/)
[![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-3.6-black.svg)](https://kafka.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Reference Paper:** Basharat Ali, Guihai Chen — *"Next-generation AI for advanced threat detection and security enhancement in DNS over HTTPS"*, **Journal of Network and Computer Applications**, Elsevier.
>
> **State Key Laboratory of Novel Software Technology, Nanjing University, Nanjing, Jiangsu 210093, China.**

---

## 📌 Overview

DNS over HTTPS (DoH) encrypts DNS queries inside HTTPS (port 443), granting privacy to end users but simultaneously creating a **blind spot** that malicious actors exploit for:

- **Command-and-Control (C2)** over encrypted channels
- **Data exfiltration** via DNS tunneling
- **Polymorphic malware delivery**
- **TLS/HTTPS session hijacking**

Classic signature-based IDS/DPI tools are rendered **useless** against encrypted DoH traffic. Existing AI-driven defenses suffer from three fatal gaps:

| Gap | Consequence |
|-----|-------------|
| Lack of adversarial robustness | Single-pixel/feature perturbation bypasses detection |
| Lack of explainability | Black-box alerts → no forensic accountability |
| Lack of real-time scalability | 180–220 ms batch latency → unusable at line rate |

**Next-GenAI** bridges all three gaps with a hybrid AI ensemble + secure execution + streaming pipeline.

---

## 🧠 Core Architecture (Fig. 1 of the paper)
