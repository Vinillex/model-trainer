# Class guide

What each drone class means and which pictures belong in it. Keep reviewing consistent with this over weeks. The camera sits on a drone that
flies mostly level, so the target is seen **in the air, from the air**: against sky or cloud, or against terrain below.

| Class | Means | Not |
|---|---|---|
| `multirotor` | quadcopters, hexacopters and other rotor-lift drones | fixed-wing, delta-wing, interceptors |
| `fixed-wing` | conventional aeroplane layout: fuselage + wing + tail plane (Bayraktar TB2/TB3/Akinci, Orlan-10, Lancet, Wing Loong, small RC planes) | delta or flying wings |
| `delta-wing` | tailless delta or flying-wing planform (no separate horizontal tail). Renamed from `shahed` on 2026-10-25: a whole family, not one aircraft | anything with a tail plane |
| `interceptor` | ramming quadcopters and FPV drones built to hit other drones (Sting, Bullet, Anvil style) | ordinary photo/video quadcopters (those are `multirotor`) |

## Picture tiers (applied while reviewing)

- **A, wanted:** air-to-air, chase-plane, gun-camera or interceptor-POV shots, aerial pictures level or from above. Target airborne, camera airborne.
- **B, keep but limited (about 40% of a class at most):** a person on the ground looking up at a target in flight against the sky.
- **C, reject (Trash):** on the ground, launch rail, runway or take-off, parade, factory, wreckage, close-up product shots, cockpit, schematics, artwork, renders.

## The delta-wing family

"check" means the planform is not confirmed; verify it when its pictures are collected.

- **Core, one-way attack and decoy delta wings:** Shahed-136 = Geran-2; Shahed-131 = Geran-1; Shahed-238 = Geran-3 (jet); Arash-2 (Iran); Samad-3 (Houthi, check);
  Gerbera and Parodiya decoys (Russia, check); LUCAS (USA, Shahed-derived); Sunflower-200 (China); Mudamir-LR/MR and HighMark-25 (Pakistan);
  Sheshnaag-150 (India); IAI Harpy and Harop (Israel) and Chinese clones (ASN-301); Italmas (Russia, check).
- **Flying-wing combat and ISR UAVs:** TAI Anka-3 (Turkey); Baykar Kizilelma (tailless with canards, check); Northrop X-47B; RQ-170 Sentinel; nEUROn;
  BAE Taranis; Sukhoi S-70 Okhotnik; GJ-11 (China); MQ-28 Ghost Bat (check).
- **Secondary, look-alikes in the air (clear in-flight pictures only, at most a quarter of the class):** senseFly eBee X, Parrot Disco, Skywalker X8, Zohd Dart
  and similar foam or RC delta and flying wings; Fire Point FP-1 (sources disagree whether it is delta, check).

## Old names

Dataset versions and trained models made before 2026-10-25 keep the name `shahed` in their own frozen labels, on purpose: they are snapshots.
Renaming a class (Overview, Rename) moves its images' files to the folder of the new name and leaves existing versions downloadable.
