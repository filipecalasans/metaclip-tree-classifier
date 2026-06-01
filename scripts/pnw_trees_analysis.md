# Pacific Northwest Tree Species — MetaCLIP Embedding Analysis

**Model:** MetaCLIP ViT-B-32 (`metaclip_400m`)
**Prompt template:** `"a photo of a {} tree"`
**Date:** 2026-05-31

---

## Species List (28 total)

### Conifers (17)

| Species | Notes |
|---|---|
| Douglas fir | Dominant PNW lowland/montane conifer |
| western red cedar | Iconic coastal and riparian conifer |
| Sitka spruce | Coastal rainforest species |
| western hemlock | Shade-tolerant climax species |
| grand fir | Low-elevation fir, moist sites |
| subalpine fir | High-elevation fir |
| Pacific silver fir | Mid-to-high elevation fir |
| lodgepole pine | Dry interior and subalpine |
| western white pine | Mixed conifer forests |
| ponderosa pine | Dry east-side forests |
| sugar pine | Southern Cascades, largest pine cones |
| whitebark pine | High alpine, keystone species |
| mountain hemlock | Subalpine, often with Pacific silver fir |
| Alaska yellow cedar | Coastal and subalpine bogs |
| Port Orford cedar | Southwest Oregon endemic |
| Pacific yew | Understory conifer, taxol source |
| western larch | Deciduous conifer, east Cascades |

### Broadleaf / Deciduous (11)

| Species | Notes |
|---|---|
| bigleaf maple | Most common PNW hardwood |
| vine maple | Understory shrub/small tree |
| red alder | Pioneer riparian species |
| black cottonwood | Largest North American poplar |
| quaking aspen | Clonal groves, east Cascades |
| Pacific madrone | Evergreen broadleaf, coastal |
| Oregon white oak | Only native oak in PNW |
| bitter cherry | Pioneer small tree |
| cascara | Riparian shrub/small tree |
| Pacific dogwood | Understory flowering tree |
| Oregon ash | Wet lowland sites |
| black hawthorn | Riparian shrub/small tree |

---

## Pairwise Cosine Similarity Matrix

Similarity is computed between L2-normalised text embeddings. Values range from 0 (orthogonal) to 1 (identical). The diagonal (self-similarity) is always 1.000.

```
                        Douglas fi  western re  Sitka spru  western he   grand fir  subalpine   Pacific si  lodgepole   western wh  ponderosa   sugar pine  whitebark   mountain h  Alaska yel  Port Orfor  Pacific ye  western la  bigleaf ma  vine maple   red alder  black cott  quaking as  Pacific ma  Oregon whi  bitter che     cascara  Pacific do  Oregon ash  black hawt
Douglas fir                  1.000       0.873       0.907       0.885       0.940       0.930       0.926       0.860       0.871       0.859       0.889       0.864       0.874       0.826       0.842       0.769       0.825       0.804       0.735       0.755       0.759       0.708       0.770       0.771       0.727       0.696       0.681       0.800       0.673
western red cedar            0.873       1.000       0.826       0.867       0.841       0.841       0.834       0.812       0.861       0.794       0.846       0.835       0.834       0.904       0.884       0.785       0.819       0.821       0.761       0.761       0.791       0.697       0.778       0.812       0.742       0.714       0.675       0.816       0.693
Sitka spruce                 0.907       0.826       1.000       0.844       0.898       0.907       0.902       0.833       0.864       0.833       0.887       0.853       0.831       0.818       0.828       0.758       0.805       0.792       0.756       0.753       0.735       0.698       0.763       0.753       0.737       0.710       0.689       0.769       0.692
western hemlock              0.885       0.867       0.844       1.000       0.843       0.849       0.844       0.801       0.850       0.789       0.831       0.831       0.969       0.815       0.818       0.772       0.804       0.780       0.729       0.744       0.740       0.687       0.748       0.760       0.717       0.673       0.690       0.797       0.702
grand fir                    0.940       0.841       0.898       0.843       1.000       0.939       0.936       0.835       0.855       0.839       0.893       0.845       0.846       0.800       0.796       0.782       0.816       0.799       0.733       0.742       0.749       0.687       0.757       0.759       0.740       0.703       0.644       0.775       0.675
subalpine fir                0.930       0.841       0.907       0.849       0.939       1.000       0.928       0.879       0.880       0.870       0.912       0.882       0.855       0.799       0.795       0.758       0.816       0.778       0.719       0.718       0.744       0.714       0.755       0.721       0.722       0.684       0.642       0.757       0.664
Pacific silver fir           0.926       0.834       0.902       0.844       0.936       0.928       1.000       0.851       0.885       0.847       0.901       0.855       0.838       0.805       0.807       0.782       0.796       0.784       0.725       0.734       0.740       0.699       0.786       0.779       0.716       0.682       0.686       0.784       0.674
lodgepole pine               0.860       0.812       0.833       0.801       0.835       0.879       0.851       1.000       0.898       0.909       0.900       0.901       0.791       0.797       0.784       0.735       0.846       0.771       0.710       0.720       0.761       0.777       0.752       0.728       0.709       0.656       0.643       0.778       0.679
western white pine           0.871       0.861       0.864       0.850       0.855       0.880       0.885       0.898       1.000       0.895       0.916       0.911       0.816       0.819       0.810       0.739       0.838       0.779       0.734       0.728       0.765       0.729       0.751       0.772       0.734       0.648       0.680       0.777       0.710
ponderosa pine               0.859       0.794       0.833       0.789       0.839       0.870       0.847       0.909       0.895       1.000       0.886       0.893       0.777       0.776       0.768       0.702       0.798       0.753       0.679       0.719       0.771       0.739       0.772       0.743       0.707       0.657       0.662       0.772       0.690
sugar pine                   0.889       0.846       0.887       0.831       0.893       0.912       0.901       0.900       0.916       0.886       1.000       0.879       0.821       0.814       0.800       0.777       0.829       0.827       0.763       0.754       0.797       0.745       0.793       0.757       0.800       0.720       0.710       0.804       0.732
whitebark pine               0.864       0.835       0.853       0.831       0.845       0.882       0.855       0.901       0.911       0.893       0.879       1.000       0.824       0.808       0.784       0.726       0.838       0.785       0.723       0.752       0.793       0.784       0.773       0.762       0.738       0.636       0.655       0.783       0.716
mountain hemlock             0.874       0.834       0.831       0.969       0.846       0.855       0.838       0.791       0.816       0.777       0.821       0.824       1.000       0.794       0.803       0.784       0.791       0.785       0.727       0.737       0.736       0.680       0.749       0.742       0.717       0.673       0.684       0.772       0.717
Alaska yellow cedar          0.826       0.904       0.818       0.815       0.800       0.799       0.805       0.797       0.819       0.776       0.814       0.808       0.794       1.000       0.868       0.743       0.808       0.808       0.749       0.736       0.784       0.729       0.770       0.805       0.719       0.676       0.669       0.817       0.673
Port Orford cedar            0.842       0.884       0.828       0.818       0.796       0.795       0.807       0.784       0.810       0.768       0.800       0.784       0.803       0.868       1.000       0.756       0.770       0.764       0.727       0.722       0.744       0.666       0.774       0.769       0.683       0.679       0.669       0.771       0.662
Pacific yew                  0.769       0.785       0.758       0.772       0.782       0.758       0.782       0.735       0.739       0.702       0.777       0.726       0.784       0.743       0.756       1.000       0.748       0.742       0.696       0.670       0.661       0.580       0.732       0.691       0.684       0.681       0.633       0.709       0.679
western larch                0.825       0.819       0.805       0.804       0.816       0.816       0.796       0.846       0.838       0.798       0.829       0.838       0.791       0.808       0.770       0.748       1.000       0.797       0.743       0.753       0.768       0.756       0.706       0.724       0.740       0.632       0.642       0.773       0.719
bigleaf maple                0.804       0.821       0.792       0.780       0.799       0.778       0.784       0.771       0.779       0.753       0.827       0.785       0.785       0.808       0.764       0.742       0.797       1.000       0.904       0.827       0.845       0.779       0.819       0.831       0.815       0.741       0.761       0.864       0.756
vine maple                   0.735       0.761       0.756       0.729       0.733       0.719       0.725       0.710       0.734       0.679       0.763       0.723       0.727       0.749       0.727       0.696       0.743       0.904       1.000       0.798       0.795       0.759       0.772       0.769       0.800       0.700       0.758       0.824       0.756
red alder                    0.755       0.761       0.753       0.744       0.742       0.718       0.734       0.720       0.728       0.719       0.754       0.752       0.737       0.736       0.722       0.670       0.753       0.827       0.798       1.000       0.799       0.789       0.774       0.796       0.800       0.648       0.734       0.846       0.737
black cottonwood             0.759       0.791       0.735       0.740       0.749       0.744       0.740       0.761       0.765       0.771       0.797       0.793       0.736       0.784       0.744       0.661       0.768       0.845       0.795       0.799       1.000       0.803       0.788       0.786       0.826       0.670       0.719       0.839       0.834
quaking aspen                0.708       0.697       0.698       0.687       0.687       0.714       0.699       0.777       0.729       0.739       0.745       0.784       0.680       0.729       0.666       0.580       0.756       0.779       0.759       0.789       0.803       1.000       0.722       0.720       0.725       0.588       0.673       0.818       0.690
Pacific madrone              0.770       0.778       0.763       0.748       0.757       0.755       0.786       0.752       0.751       0.772       0.793       0.773       0.749       0.770       0.774       0.732       0.706       0.819       0.772       0.774       0.788       0.722       1.000       0.795       0.761       0.696       0.783       0.821       0.745
Oregon white oak             0.771       0.812       0.753       0.760       0.759       0.721       0.779       0.728       0.772       0.743       0.757       0.762       0.742       0.805       0.769       0.691       0.724       0.831       0.769       0.796       0.786       0.720       0.795       1.000       0.748       0.672       0.694       0.880       0.710
bitter cherry                0.727       0.742       0.737       0.717       0.740       0.722       0.716       0.709       0.734       0.707       0.800       0.738       0.717       0.719       0.683       0.684       0.740       0.815       0.800       0.800       0.826       0.725       0.761       0.748       1.000       0.698       0.778       0.812       0.823
cascara                      0.696       0.714       0.710       0.673       0.703       0.684       0.682       0.656       0.648       0.657       0.720       0.636       0.673       0.676       0.679       0.681       0.632       0.741       0.700       0.648       0.670       0.588       0.696       0.672       0.698       1.000       0.615       0.697       0.664
Pacific dogwood              0.681       0.675       0.689       0.690       0.644       0.642       0.686       0.643       0.680       0.662       0.710       0.655       0.684       0.669       0.669       0.633       0.642       0.761       0.758       0.734       0.719       0.673       0.783       0.694       0.778       0.615       1.000       0.760       0.707
Oregon ash                   0.800       0.816       0.769       0.797       0.775       0.757       0.784       0.778       0.777       0.772       0.804       0.783       0.772       0.817       0.771       0.709       0.773       0.864       0.824       0.846       0.839       0.818       0.821       0.880       0.812       0.697       0.760       1.000       0.756
black hawthorn               0.673       0.693       0.692       0.702       0.675       0.664       0.674       0.679       0.710       0.690       0.732       0.716       0.717       0.673       0.662       0.679       0.719       0.756       0.756       0.737       0.834       0.690       0.745       0.710       0.823       0.664       0.707       0.756       1.000
```

---

## High-Confusion Pairs (similarity >= 0.90)

| Score | Species A | Species B |
|---|---|---|
| 0.969 | western hemlock | mountain hemlock |
| 0.940 | Douglas fir | grand fir |
| 0.939 | grand fir | subalpine fir |
| 0.936 | grand fir | Pacific silver fir |
| 0.930 | Douglas fir | subalpine fir |
| 0.928 | subalpine fir | Pacific silver fir |
| 0.926 | Douglas fir | Pacific silver fir |
| 0.916 | western white pine | sugar pine |
| 0.912 | subalpine fir | sugar pine |
| 0.911 | western white pine | whitebark pine |
| 0.909 | lodgepole pine | ponderosa pine |
| 0.907 | Douglas fir | Sitka spruce |
| 0.907 | Sitka spruce | subalpine fir |
| 0.904 | bigleaf maple | vine maple |
| 0.904 | western red cedar | Alaska yellow cedar |
| 0.902 | Sitka spruce | Pacific silver fir |
| 0.901 | lodgepole pine | whitebark pine |
| 0.901 | Pacific silver fir | sugar pine |

---

## Weakest Representations

Species with the lowest mean cosine similarity to all others. These labels are the least grounded in the model's pretraining data and are most likely to underperform at inference time.

| Mean similarity | Species |
|---|---|
| 0.675 | cascara |
| 0.691 | Pacific dogwood |
| 0.712 | black hawthorn |
| 0.719 | quaking aspen |
| 0.724 | Pacific yew |

---

## Recommendations

### Confused species groups

Three dense confusion clusters exist in the conifer list:

**Fir cluster** — all pairs score 0.926–0.940:
- Douglas fir, grand fir, subalpine fir, Pacific silver fir

**Pine cluster** — all pairs score 0.900–0.916:
- western white pine, sugar pine, whitebark pine, lodgepole pine, ponderosa pine

**Hemlock pair** — highest confusion in the entire list at 0.969:
- western hemlock, mountain hemlock

**Cedar pair** — 0.904:
- western red cedar, Alaska yellow cedar

### Suggested mitigations

| Issue | Action |
|---|---|
| western hemlock / mountain hemlock (0.969) | Drop one, or use habitat prompts: `"western hemlock in coastal rainforest"` vs `"mountain hemlock at subalpine treeline"` |
| Fir cluster (0.926–0.940) | Add elevation cues: `"grand fir in lowland mixed forest"`, `"subalpine fir at high elevation"` |
| Pine cluster (0.900–0.916) | Add habitat cues: `"ponderosa pine in dry open forest"`, `"whitebark pine at alpine treeline"` |
| cascara (0.675) | Rename to `"cascara buckthorn"` for a more grounded embedding |
| Pacific dogwood (0.691) | Try `"Cornus nuttallii"` or `"Pacific dogwood flowering tree"` |
| black hawthorn (0.712) | Try `"Crataegus douglasii"` or add `"thorny"` descriptor |

---

## Reproduction

```bash
source activate.sh
python scripts/pairwise_similarity.py
```
