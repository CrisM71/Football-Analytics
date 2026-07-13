# 🇯🇵 Japan vs. Spain 🇪🇸 | World Cup 2022 Advanced Tactical Analytics

An simple and first sports analytics project evaluating one of the most statistically fascinating matches in FIFA World Cup history: Japan's 2-1 victory over Spain in 2022. Using Python, **StatsBomb's Open Data**, and **mplsoccer**, this repository builds data-driven pipelines to explore the extreme tactical clash between ultra-efficient low-block transitions and high-volume positional possession (Tiki-Taka).

---

## 📊 Visual Insights & Mathematical Models

### 1. Shot Map (Contextual xG)
This visualization splits the pitch to display each team's shooting efficiency. Instead of generic markers, this pipeline categorizes attempts by their physical outcome to expose defensive density:
*   **Stars:** Successful goals.
*   **Circles:** Saved, missed, or post-striking shots.
*   **Triangles:** Shots physically blocked by the opponent's defensive wall.

![Shot Map](https://raw.githubusercontent.com/CrisM71/Football-Analytics/main/output/shot_map.png)

*   **Tactical Insight:** Spain dominated raw shot volume (12 vs. 6) but struggled with shot quality due to Japan's compact spatial control. Spain's map is crowded with **blocked triangles**, leading to a low average xG per shot (0.07). Conversely, Japan's structured counter-attacks created highly optimized finishing angles, yielding a massive 0.19 average xG per shot and scoring two goals from just 1.16 total xG.

### 2. Positional Passing Networks & Expected Threat (xT)
Calculated exclusively for the **initial tactical period** (pre-minute 45, before substitution biases dilute structural integrity) with automatic error protection for Spanish nickname mappings (e.g., standardizing *Pedri, Gavi, Rodri, Busquets*).

This model integrates **Karun Singh's original 12x8 Expected Threat (xT) spatial probability matrix** via Markov chains:
*   **Node Size:** Scales dynamically based on the player's **Cumulative xT** (real threat generated via progressive actions), rather than raw passing volume.
*   **Line Weight & Opacity:** The width represents pass frequency, while the brightness/opacity highlights the **Threat Delta ($\Delta xT$)** of the combination.

![Pass Network and xT](output/pass_network.png)

*   **Tactical Insight:** Spain’s network forms a suffocatingly dense web of short, horizontal circulation around the midfield line with low xT generation per pass. Japan’s network exposes a deep, resilient defensive diamond. Their front line remains entirely isolated from short-passing build-up, mathematically validating a pure, direct vertical transition framework.

---

## 🛠️ Tech Stack & Architecture

*   **Language:** Python 3.13
*   **Core Libraries:** `pandas`, `numpy`, `matplotlib`, `mplsoccer`, `statsbombpy`
*   **Algorithmic Concept:** Karun Singh's Expected Threat ($xT$) Spatial Bilinear Interpolation

### Engineering Pipeline Highlights
1.  **Robust Ingestion Client:** Automated event extraction via StatsBomb's API client with programmatic schema validation to handle empty unassigned columns (`player_nickname` protection).
2.  **Tactical Phase Filtering:** Programmatic isolation of the pre-substitution window to map clean tactical starting frameworks.
3.  **Advanced Geometry Mapping:** Custom Python functions to translate continuous event tracking coordinate arrays ($120 \times 80$) into discrete matrix grids for $xT$ indexing and visualization.

---

## 🧑‍💻 Author

**Cristiano Santos**  
*B.Sc. in Informatics Engineering (University of Coimbra)*  
*M.Sc. Student in Artificial Intelligence — Focusing on AI Applications in Sports Analytics*

*   **LinkedIn:** [https://www.linkedin.com/in/cristiano-santos-b173402a1]
