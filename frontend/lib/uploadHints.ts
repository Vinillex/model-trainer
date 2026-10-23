/** A cheap guess at whether a folder's name and the class it is uploaded to disagree. Only a hint: the person decides.
 *  It matters because images uploaded into a negative class arrive already reviewed, as background: a folder of drones put there would be trained on as "no drone". */
const DRONE = /drone|uav|quad|multirotor|copter|shahed|vtol|fixed[-_ .]?wing|interceptor/i;
const NOT_DRONE = /bird|balloon|aircraft|airplane|plane|helicopter|cloud|sky|negative|background|empty|kite|insect|smoke|fog/i;

export function folderHint(folder: string, className: string, negative: boolean): string | null {
  if (negative && DRONE.test(folder) && !NOT_DRONE.test(folder)) {
    return `“${folder}” looks like drone images, but “${className}” is a hard-negative class: its images count as background (no drones) and go into training as already reviewed.`;
  }
  if (!negative && NOT_DRONE.test(folder) && !DRONE.test(folder)) {
    return `“${folder}” looks like it has no drones in it, but “${className}” is a drone class: you would have to review every image and remove the boxes.`;
  }
  return null;
}
