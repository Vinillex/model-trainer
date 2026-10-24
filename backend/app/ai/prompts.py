"""The words the text-prompt detector (app/ai/yoloe.py) is given. Defaults only: the person can type their own in the AI panel.

Chosen by measuring, not by taste (2026-10-24, 886 labelled drone images, scripts in HANDOVER 4.8): YOLOE finds a drone with words like
"drone aircraft" (90% of drones of 32+ px at score 0.10, 82% at 0.25, 25% of the smaller ones), but it cannot tell our classes apart by name:
"quadcopter drone" found 8% of the multirotors and called most of them "fixed-wing" or "interceptor". So one generic phrase finds the drones
and the class comes from the folder an image is filed under, which the person checks in review anyway."""

DRONE_PROMPT = "drone aircraft"
DRONE_LABEL = "drone"  # what the detector calls a hit on the drone phrase (never a project class name)
OBJECT_SCORE = 0.25  # a detection of the negative class's own thing ("bird") counts from this score (bird sizes it gave matched a by-eye reading)

OBJECT_DEFAULTS = {"birds": "bird", "balloons": "balloon", "aircraft": "airplane", "helicopters": "helicopter"}


def object_label(class_id: int) -> str:
    """The detector's label for the thing a negative class is about; ids, because class names may collide with DRONE_LABEL."""
    return f"class:{class_id}"


def default_object_prompt(name: str) -> str:
    """Words for what a negative class holds, so the run can check its size. Empty for a class with nothing to point at ("empty-sky", "smoke")."""
    return OBJECT_DEFAULTS.get(name.lower(), "")
