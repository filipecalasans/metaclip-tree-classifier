# Pacific Northwest tree species with habitat-aware prompts.
# Keys are display names; values are the full prompts passed to the text encoder.
PNW_TREES: dict[str, str] = {
    # Conifers
    "Douglas fir":         "a photo of a Douglas fir tree in a Pacific Northwest lowland mixed forest",
    "western red cedar":   "a photo of a western red cedar tree in a coastal rainforest with hanging moss",
    "Sitka spruce":        "a photo of a Sitka spruce tree in a coastal fog belt rainforest",
    "western hemlock":     "a photo of a western hemlock tree in a shaded coastal rainforest understory",
    "grand fir":           "a photo of a grand fir tree in a low-elevation moist mixed forest",
    "subalpine fir":       "a photo of a subalpine fir tree at high-elevation treeline",
    "Pacific silver fir":  "a photo of a Pacific silver fir tree in a mid-elevation Cascade montane forest",
    "lodgepole pine":      "a photo of a lodgepole pine tree in a dry subalpine or post-fire forest",
    "western white pine":  "a photo of a western white pine tree in a mixed conifer forest",
    "ponderosa pine":      "a photo of a ponderosa pine tree in an open dry east-side forest",
    "sugar pine":          "a photo of a sugar pine tree in a southern Cascade mixed conifer forest with large cones",
    "whitebark pine":      "a photo of a whitebark pine tree at an alpine treeline",
    "mountain hemlock":    "a photo of a mountain hemlock tree at a subalpine treeline with heavy snow",
    "Alaska yellow cedar": "a photo of an Alaska yellow cedar tree in a coastal subalpine bog",
    "Port Orford cedar":   "a photo of a Port Orford cedar tree in a southwest Oregon riparian forest",
    "Pacific yew":         "a photo of a Pacific yew tree as a shaded forest understory conifer",
    "western larch":       "a photo of a western larch deciduous conifer tree in an east Cascade dry forest",
    # Broadleaf / deciduous
    "bigleaf maple":       "a photo of a bigleaf maple tree with large leaves in a Pacific Northwest riparian forest",
    "vine maple":          "a photo of a vine maple small tree in a shaded forest understory",
    "red alder":           "a photo of a red alder tree along a Pacific Northwest stream or riparian area",
    "black cottonwood":    "a photo of a black cottonwood tree along a river floodplain",
    "quaking aspen":       "a photo of a quaking aspen tree in a clonal grove on the east side of the Cascades",
    "Pacific madrone":     "a photo of a Pacific madrone evergreen tree with red peeling bark on a coastal bluff",
    "Oregon white oak":    "a photo of an Oregon white oak tree in a dry savanna or woodland",
    "bitter cherry":       "a photo of a bitter cherry small tree in a disturbed or forest edge habitat",
    "cascara":             "a photo of a cascara buckthorn small tree in a Pacific Northwest riparian woodland",
    "Pacific dogwood":     "a photo of a Pacific dogwood flowering tree in a shaded forest understory",
    "Oregon ash":          "a photo of an Oregon ash tree in a wet lowland forest or floodplain",
    "black hawthorn":      "a photo of a black hawthorn thorny shrub tree along a Pacific Northwest stream",
}

# Generic plant species list for general-purpose classification.
# Extend or replace with a domain-specific list as needed.
PLANT_SPECIES = [
    # Trees
    "oak tree",
    "maple tree",
    "pine tree",
    "birch tree",
    "cherry blossom tree",
    "eucalyptus tree",
    "palm tree",
    "willow tree",
    "cedar tree",
    "redwood tree",
    # Flowering plants
    "rose",
    "sunflower",
    "tulip",
    "daisy",
    "lavender",
    "orchid",
    "lily",
    "dandelion",
    "poppy",
    "marigold",
    "hydrangea",
    "peony",
    "iris",
    "hibiscus",
    "jasmine",
    # Succulents and cacti
    "aloe vera",
    "cactus",
    "agave",
    "jade plant",
    "echeveria",
    # Ferns and mosses
    "fern",
    "moss",
    "horsetail",
    # Herbs
    "basil",
    "mint",
    "rosemary",
    "thyme",
    "sage",
    # Vegetables (plant form)
    "tomato plant",
    "pepper plant",
    "lettuce",
    "spinach",
    "broccoli plant",
    # Aquatic plants
    "water lily",
    "lotus flower",
    "cattail",
    # Vines and climbers
    "ivy",
    "bougainvillea",
    "wisteria",
]
