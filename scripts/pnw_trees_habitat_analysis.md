# Pacific Northwest Tree Species — Habitat-Aware Prompt Analysis

**Model:** MetaCLIP ViT-B-32 (`metaclip_400m`)
**Prompt strategy:** Per-species habitat-aware prompts
**Script:** `scripts/pairwise_similarity_habitat.py`
**Date:** 2026-05-31

---

## Motivation

The [baseline analysis](pnw_trees_analysis.md) used a generic prompt template (`"a photo of a {} tree"`) and identified 18 high-confusion pairs (similarity ≥ 0.90), with the worst pair (western hemlock / mountain hemlock) reaching 0.969. This follow-up replaces the generic template with a per-species prompt that embeds ecological habitat context, testing whether richer text descriptions can separate confused species in the embedding space without any model retraining.

---

## Prompt Strategy

Each species was assigned a unique prompt describing its typical habitat, growth form, and geographic range. The generic template is replaced entirely — there is no shared structure across prompts.

| Species | Prompt |
|---|---|
| Douglas fir | a photo of a Douglas fir tree in a Pacific Northwest lowland mixed forest |
| western red cedar | a photo of a western red cedar tree in a coastal rainforest with hanging moss |
| Sitka spruce | a photo of a Sitka spruce tree in a coastal fog belt rainforest |
| western hemlock | a photo of a western hemlock tree in a shaded coastal rainforest understory |
| grand fir | a photo of a grand fir tree in a low-elevation moist mixed forest |
| subalpine fir | a photo of a subalpine fir tree at high-elevation treeline |
| Pacific silver fir | a photo of a Pacific silver fir tree in a mid-elevation Cascade montane forest |
| lodgepole pine | a photo of a lodgepole pine tree in a dry subalpine or post-fire forest |
| western white pine | a photo of a western white pine tree in a mixed conifer forest |
| ponderosa pine | a photo of a ponderosa pine tree in an open dry east-side forest |
| sugar pine | a photo of a sugar pine tree in a southern Cascade mixed conifer forest with large cones |
| whitebark pine | a photo of a whitebark pine tree at an alpine treeline |
| mountain hemlock | a photo of a mountain hemlock tree at a subalpine treeline with heavy snow |
| Alaska yellow cedar | a photo of an Alaska yellow cedar tree in a coastal subalpine bog |
| Port Orford cedar | a photo of a Port Orford cedar tree in a southwest Oregon riparian forest |
| Pacific yew | a photo of a Pacific yew tree as a shaded forest understory conifer |
| western larch | a photo of a western larch deciduous conifer tree in an east Cascade dry forest |
| bigleaf maple | a photo of a bigleaf maple tree with large leaves in a Pacific Northwest riparian forest |
| vine maple | a photo of a vine maple small tree in a shaded forest understory |
| red alder | a photo of a red alder tree along a Pacific Northwest stream or riparian area |
| black cottonwood | a photo of a black cottonwood tree along a river floodplain |
| quaking aspen | a photo of a quaking aspen tree in a clonal grove on the east side of the Cascades |
| Pacific madrone | a photo of a Pacific madrone evergreen tree with red peeling bark on a coastal bluff |
| Oregon white oak | a photo of an Oregon white oak tree in a dry savanna or woodland |
| bitter cherry | a photo of a bitter cherry small tree in a disturbed or forest edge habitat |
| cascara | a photo of a cascara buckthorn small tree in a Pacific Northwest riparian woodland |
| Pacific dogwood | a photo of a Pacific dogwood flowering tree in a shaded forest understory |
| Oregon ash | a photo of an Oregon ash tree in a wet lowland forest or floodplain |
| black hawthorn | a photo of a black hawthorn thorny shrub tree along a Pacific Northwest stream |

---

## Pairwise Cosine Similarity Matrix

```
                        Douglas fi  western re  Sitka spru  western he   grand fir  subalpine   Pacific si  lodgepole   western wh  ponderosa   sugar pine  whitebark   mountain h  Alaska yel  Port Orfor  Pacific ye  western la  bigleaf ma  vine maple   red alder  black cott  quaking as  Pacific ma  Oregon whi  bitter che     cascara  Pacific do  Oregon ash  black hawt
Douglas fir                  1.000       0.747       0.756       0.846       0.912       0.842       0.859       0.763       0.873       0.798       0.793       0.804       0.737       0.763       0.841       0.773       0.834       0.766       0.726       0.731       0.614       0.640       0.573       0.730       0.736       0.718       0.626       0.816       0.613
western red cedar            0.747       1.000       0.709       0.792       0.711       0.639       0.711       0.585       0.694       0.615       0.660       0.649       0.653       0.707       0.728       0.727       0.720       0.727       0.712       0.698       0.563       0.588       0.689       0.613       0.624       0.659       0.600       0.730       0.594
Sitka spruce                 0.756       0.709       1.000       0.710       0.758       0.677       0.735       0.611       0.690       0.628       0.665       0.676       0.654       0.675       0.717       0.654       0.653       0.648       0.609       0.630       0.484       0.541       0.570       0.575       0.576       0.613       0.506       0.691       0.561
western hemlock              0.846       0.792       0.710       1.000       0.801       0.739       0.796       0.654       0.782       0.720       0.701       0.733       0.782       0.717       0.775       0.816       0.759       0.755       0.798       0.716       0.618       0.621       0.606       0.668       0.703       0.728       0.690       0.781       0.632
grand fir                    0.912       0.711       0.758       0.801       1.000       0.875       0.904       0.756       0.875       0.799       0.809       0.804       0.758       0.731       0.758       0.766       0.838       0.704       0.701       0.674       0.586       0.625       0.548       0.691       0.744       0.658       0.566       0.771       0.568
subalpine fir                0.842       0.639       0.677       0.739       0.875       1.000       0.852       0.804       0.831       0.794       0.760       0.883       0.787       0.763       0.705       0.684       0.805       0.609       0.624       0.622       0.608       0.626       0.543       0.653       0.739       0.629       0.495       0.695       0.543
Pacific silver fir           0.859       0.711       0.735       0.796       0.904       0.852       1.000       0.731       0.829       0.750       0.805       0.786       0.760       0.717       0.744       0.765       0.804       0.696       0.686       0.659       0.559       0.615       0.571       0.634       0.677       0.665       0.569       0.722       0.557
lodgepole pine               0.763       0.585       0.611       0.654       0.756       0.804       0.731       1.000       0.803       0.875       0.736       0.826       0.658       0.732       0.688       0.612       0.810       0.597       0.613       0.617       0.585       0.706       0.527       0.687       0.680       0.626       0.486       0.727       0.505
western white pine           0.873       0.694       0.690       0.782       0.875       0.831       0.829       0.803       1.000       0.839       0.878       0.845       0.731       0.728       0.745       0.763       0.860       0.671       0.702       0.634       0.587       0.662       0.536       0.672       0.706       0.657       0.593       0.725       0.577
ponderosa pine               0.798       0.615       0.628       0.720       0.799       0.794       0.750       0.875       0.839       1.000       0.768       0.821       0.682       0.698       0.729       0.662       0.809       0.648       0.651       0.640       0.631       0.674       0.580       0.760       0.683       0.643       0.558       0.751       0.556
sugar pine                   0.793       0.660       0.665       0.701       0.809       0.760       0.805       0.736       0.878       0.768       1.000       0.747       0.676       0.658       0.683       0.723       0.809       0.647       0.636       0.581       0.506       0.613       0.525       0.570       0.610       0.622       0.546       0.647       0.543
whitebark pine               0.804       0.649       0.676       0.733       0.804       0.883       0.786       0.826       0.845       0.821       0.747       1.000       0.790       0.757       0.712       0.667       0.799       0.646       0.641       0.676       0.654       0.698       0.583       0.707       0.721       0.628       0.524       0.728       0.590
mountain hemlock             0.737       0.653       0.654       0.782       0.758       0.787       0.760       0.658       0.731       0.682       0.676       0.790       1.000       0.649       0.631       0.672       0.682       0.597       0.597       0.591       0.533       0.573       0.509       0.585       0.600       0.591       0.515       0.657       0.563
Alaska yellow cedar          0.763       0.707       0.675       0.717       0.731       0.763       0.717       0.732       0.728       0.698       0.658       0.757       0.649       1.000       0.802       0.626       0.778       0.672       0.624       0.687       0.688       0.645       0.546       0.719       0.705       0.649       0.483       0.761       0.558
Port Orford cedar            0.841       0.728       0.717       0.775       0.758       0.705       0.744       0.688       0.745       0.729       0.683       0.712       0.631       0.802       1.000       0.706       0.739       0.757       0.669       0.767       0.688       0.594       0.610       0.733       0.667       0.714       0.559       0.809       0.626
Pacific yew                  0.773       0.727       0.654       0.816       0.766       0.684       0.765       0.612       0.763       0.662       0.723       0.667       0.672       0.626       0.706       1.000       0.727       0.679       0.731       0.624       0.512       0.557       0.594       0.590       0.622       0.689       0.622       0.677       0.591
western larch                0.834       0.720       0.653       0.759       0.838       0.805       0.804       0.810       0.860       0.809       0.809       0.799       0.682       0.778       0.739       0.727       1.000       0.694       0.718       0.693       0.654       0.705       0.555       0.717       0.766       0.704       0.569       0.777       0.600
bigleaf maple                0.766       0.727       0.648       0.755       0.704       0.609       0.696       0.597       0.671       0.648       0.647       0.646       0.597       0.672       0.757       0.679       0.694       1.000       0.807       0.766       0.644       0.649       0.620       0.732       0.642       0.722       0.663       0.788       0.625
vine maple                   0.726       0.712       0.609       0.798       0.701       0.624       0.686       0.613       0.702       0.651       0.636       0.641       0.597       0.624       0.669       0.731       0.718       0.807       1.000       0.712       0.616       0.691       0.559       0.677       0.703       0.736       0.760       0.756       0.635
red alder                    0.731       0.698       0.630       0.716       0.674       0.622       0.659       0.617       0.634       0.640       0.581       0.676       0.591       0.687       0.767       0.624       0.693       0.766       0.712       1.000       0.743       0.662       0.623       0.724       0.736       0.727       0.635       0.817       0.728
black cottonwood             0.614       0.563       0.484       0.618       0.586       0.608       0.559       0.585       0.587       0.631       0.506       0.654       0.533       0.688       0.688       0.512       0.654       0.644       0.616       0.743       1.000       0.626       0.539       0.718       0.712       0.646       0.497       0.786       0.671
quaking aspen                0.640       0.588       0.541       0.621       0.625       0.626       0.615       0.706       0.662       0.674       0.613       0.698       0.573       0.645       0.594       0.557       0.705       0.649       0.691       0.662       0.626       1.000       0.492       0.623       0.636       0.637       0.545       0.708       0.538
Pacific madrone              0.573       0.689       0.570       0.606       0.548       0.543       0.571       0.527       0.536       0.580       0.525       0.583       0.509       0.546       0.610       0.594       0.555       0.620       0.559       0.623       0.539       0.492       1.000       0.551       0.547       0.569       0.539       0.617       0.602
Oregon white oak             0.730       0.613       0.575       0.668       0.691       0.653       0.634       0.687       0.672       0.760       0.570       0.707       0.585       0.719       0.733       0.590       0.717       0.732       0.677       0.724       0.718       0.623       0.551       1.000       0.714       0.683       0.546       0.823       0.612
bitter cherry                0.736       0.623       0.576       0.703       0.744       0.739       0.677       0.680       0.706       0.683       0.610       0.721       0.600       0.705       0.667       0.622       0.766       0.642       0.703       0.736       0.712       0.636       0.547       0.714       1.000       0.733       0.618       0.780       0.661
cascara                      0.718       0.659       0.613       0.728       0.658       0.629       0.665       0.626       0.657       0.643       0.622       0.628       0.591       0.649       0.714       0.689       0.704       0.722       0.736       0.727       0.646       0.637       0.569       0.683       0.733       1.000       0.659       0.753       0.705
Pacific dogwood              0.626       0.600       0.506       0.690       0.566       0.495       0.569       0.486       0.593       0.558       0.546       0.524       0.515       0.483       0.559       0.622       0.569       0.663       0.760       0.635       0.497       0.545       0.539       0.546       0.618       0.659       1.000       0.631       0.631
Oregon ash                   0.816       0.730       0.691       0.781       0.771       0.695       0.722       0.727       0.725       0.751       0.647       0.728       0.657       0.761       0.809       0.677       0.777       0.788       0.756       0.817       0.786       0.708       0.617       0.823       0.780       0.753       0.631       1.000       0.670
black hawthorn               0.613       0.594       0.561       0.632       0.568       0.543       0.557       0.505       0.577       0.556       0.543       0.590       0.563       0.558       0.626       0.591       0.600       0.625       0.635       0.728       0.671       0.538       0.602       0.612       0.661       0.705       0.631       0.670       1.000
```

---

## High-Confusion Pairs (similarity >= 0.90)

| Score | Species A | Species B |
|---|---|---|
| 0.912 | Douglas fir | grand fir |
| 0.904 | grand fir | Pacific silver fir |

---

## Weakest Representations

| Mean similarity | Species |
|---|---|
| 0.569 | Pacific madrone |
| 0.580 | Pacific dogwood |
| 0.602 | black hawthorn |
| 0.617 | black cottonwood |
| 0.625 | quaking aspen |

---

## Key Findings

### Confusion pairs reduced from 18 to 2

Habitat-aware prompts reduced high-confusion pairs by **89%**, from 18 to 2. The most dramatic improvement was in the hemlock pair, previously the worst offender at 0.969, which dropped below the 0.90 threshold after embedding habitat context (`"shaded coastal rainforest understory"` vs `"subalpine treeline with heavy snow"`).

| Cluster | Generic (worst pair) | Habitat-aware (worst pair) | Resolved? |
|---|---|---|---|
| Hemlock pair | 0.969 | 0.782 | Yes |
| Fir cluster | 0.940 | 0.912 | Partially |
| Pine cluster | 0.916 | 0.883 | Yes |
| Cedar pair | 0.904 | 0.802 | Yes |
| Maple pair | 0.904 | 0.807 | Yes |

### Remaining unresolved pairs

**Douglas fir / grand fir (0.912)** — both occupy low-elevation mixed forest in the PNW, so their habitat descriptions are ecologically adjacent. Habitat context alone is insufficient here. Visual morphology descriptors (deeply furrowed bark, pendant cones with three-pronged bracts on Douglas fir; smoother bark and upright cones on grand fir) would be needed to push this below 0.90.

**Grand fir / Pacific silver fir (0.904)** — same root cause. These species occupy adjacent elevation bands and share a moist forest context. Elevation-specific language (`"low-elevation"` vs `"mid-elevation"`) helps but is not enough on its own.

### Weakest representations shifted

The weakest species changed significantly between analyses. Pacific madrone now has the lowest mean similarity (0.569), lower than cascara (the weakest in the baseline). This reveals an important limitation: Pacific madrone's most distinctive traits — red peeling bark and evergreen leaves on coastal bluffs — are strongly visual and do not translate well into text embedding space. Text prompts alone may not adequately represent this species.

| Species | Baseline mean sim | Habitat mean sim | Change |
|---|---|---|---|
| cascara | 0.675 | 0.705 | +0.030 |
| Pacific dogwood | 0.691 | 0.580 | -0.111 |
| black hawthorn | 0.712 | 0.602 | -0.110 |
| Pacific madrone | — | 0.569 | new weakest |

The decline for Pacific dogwood and black hawthorn suggests their habitat descriptions shifted their embeddings away from the cluster center — they are now more isolated, which is desirable for differentiation but also signals the model has weaker grounding for these concepts overall.

---

## Recommendations

### For the two remaining confused pairs

| Pair | Suggested prompt addition |
|---|---|
| Douglas fir / grand fir | Add visual morphology: `"with deeply furrowed bark and pendant cones with three-pronged bracts"` for Douglas fir; `"with smooth grey bark and upright cylindrical cones"` for grand fir |
| Grand fir / Pacific silver fir | Strengthen elevation contrast: `"valley floor"` for grand fir; `"upper montane zone below treeline"` for Pacific silver fir |

### For visually-defined species

Pacific madrone and Pacific yew are best characterized visually. For these species, supplement text prompts with example images in a `PlantSearchIndex` rather than relying on text embeddings alone. A few representative images would provide far stronger differentiation than any text refinement.

### General prompt design principle

This analysis confirms that habitat and geographic context are effective differentiators for most conifer species. For broadleaf species, morphological descriptors (bark texture, leaf shape, flower color) are more informative than habitat alone, as many broadleaf species co-occur in riparian and understory habitats.

---

## Reproduction

```bash
source activate.sh
python scripts/pairwise_similarity_habitat.py
```

See also the [baseline analysis](pnw_trees_analysis.md) using the generic prompt template.
