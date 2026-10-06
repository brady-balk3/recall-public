// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Your crew: the pet roster and how each pet is drawn.
 *
 * Soft vector blobs, 12 species with 3-6 looks each, drawn as inline SVG from
 * these constants (no images, nothing fetched). Gear unlocks with keepers.
 * Pets are pure decoration: no hunger, no sadness, no guilt mechanics.
 */

export type PetKind = "cat" | "dog" | "fox" | "bunny" | "slime" | "dragon" | "ghost" | "robot" | "frog" | "duck" | "capybara" | "axolotl";
export type PetGear = "none" | "bandana" | "headset" | "cap" | "crown";

export interface PetLook {
  n: string;
  c: string;
  b: string;
  d: string;
  pat?: string;
  name: string;
  eye?: string;
  ring?: boolean;
  line?: string;
  w?: string;
}

export const PET_UNLOCK_2 = 10;
export const PET_GEAR: [PetGear, string, number][] = [["none", "None", 0], ["bandana", "Bandana", 10], ["headset", "Headset", 25], ["cap", "Cap", 50], ["crown", "Crown", 100]];

/* ---------- species: [key, label, headTop] ---------- */
export const PETS: [PetKind, string, number][] = [["cat", "Cat", 38], ["dog", "Dog", 38], ["fox", "Fox", 38], ["bunny", "Bunny", 38], ["slime", "Slime", 34], ["dragon", "Dragon", 38],
  ["ghost", "Ghost", 30], ["robot", "Robot", 38], ["frog", "Frog", 33], ["duck", "Duck", 38], ["capybara", "Capybara", 44], ["axolotl", "Axolotl", 38]];
/* ---------- looks: body, belly, detail color, pattern, default name, extras (eye, ring on dark coats, mouth line, wing) ---------- */
const DARK: Partial<PetLook> = { eye: "#1D1A22", ring: true, line: "#F2C4CF" };
export const PET_LOOKS: Record<PetKind, PetLook[]> = {
  cat: [
    { n: "Orange tabby", c: "#F4B77E", b: "#FFE3C4", d: "#C47A45", pat: "tabby", name: "Miso" },
    { n: "Black", c: "#34303A", b: "#4A4452", d: "#1E1B22", name: "Salem", ...DARK, eye: "#FFD54A", ring: false },
    { n: "Tuxedo", c: "#2F2B33", b: "#FFFFFF", d: "#1B181E", pat: "tux", name: "Oreo", eye: "#9BE36D" },
    { n: "Calico", c: "#FFF6EC", b: "#FFFFFF", d: "#3A302C", pat: "calico", name: "Patches" },
    { n: "Grey tabby", c: "#A9ADB8", b: "#E4E6EC", d: "#6E7280", pat: "tabby", name: "Smokey" },
    { n: "Siamese", c: "#F3E6D3", b: "#FFF7EC", d: "#5B4336", pat: "points", name: "Suki", eye: "#3E9BFF" },
  ],
  dog: [
    { n: "Golden", c: "#E8B46E", b: "#FFF0DC", d: "#B07A3E", name: "Biscuit" },
    { n: "Black lab", c: "#4A403B", b: "#665A53", d: "#2A2320", name: "Shadow", ...DARK },
    { n: "Dalmatian", c: "#FAFAFA", b: "#FFFFFF", d: "#26262C", pat: "spots", name: "Domino" },
    { n: "Husky", c: "#9EA7B6", b: "#FFFFFF", d: "#4E5666", pat: "mask", name: "Nova", eye: "#3EA8F0" },
    { n: "Corgi", c: "#F0A04B", b: "#FFFFFF", d: "#B96A24", pat: "blaze", name: "Waffles" },
  ],
  fox: [
    { n: "Red", c: "#F28A4C", b: "#FFF4EA", d: "#6A3A22", name: "Ember" },
    { n: "Arctic", c: "#F2F4F9", b: "#FFFFFF", d: "#AEB7CA", name: "Frost" },
    { n: "Silver", c: "#5C6070", b: "#E8EAF0", d: "#2A2C35", name: "Ash", eye: "#FFC53D", line: "#1D1A22" },
  ],
  bunny: [
    { n: "Snow", c: "#F3EDF1", b: "#FFFFFF", d: "#F4A8BF", name: "Mochi" },
    { n: "Cocoa", c: "#B98B67", b: "#EBD6C4", d: "#F4A8BF", name: "Cocoa" },
    { n: "Dutch", c: "#3A3540", b: "#FFFFFF", d: "#F4A8BF", pat: "dutch", name: "Panda", ring: true },
    { n: "Pebble", c: "#AEB0B8", b: "#E6E7EB", d: "#F4A8BF", name: "Pebble" },
  ],
  slime: [
    { n: "Mint", c: "#7FE3A8", b: "#C6F7DA", d: "#3BA56B", name: "Gloop" },
    { n: "Blue", c: "#7FC4FF", b: "#CFE8FF", d: "#3F86C9", name: "Bloop" },
    { n: "Bubblegum", c: "#FF9FD0", b: "#FFD6EA", d: "#D55C9A", name: "Jelly" },
    { n: "Honey", c: "#FFD66B", b: "#FFF0C2", d: "#C99B2A", name: "Honey" },
    { n: "Void", c: "#3B2F52", b: "#5B4B7A", d: "#1E1730", name: "Void", ...DARK, eye: "#C9B8FF", ring: false },
  ],
  dragon: [
    { n: "Lilac", c: "#9A88F2", b: "#D4CBFF", d: "#FFD27A", w: "#7C69DC", name: "Pip" },
    { n: "Ember", c: "#F0645A", b: "#FFC7A8", d: "#FFD27A", w: "#C94A42", name: "Scorch" },
    { n: "Moss", c: "#5FCB8A", b: "#D4F5C4", d: "#FFE08A", w: "#3E9C66", name: "Sprout" },
    { n: "Glacier", c: "#8FD8F0", b: "#E5F8FF", d: "#FFFFFF", w: "#5AAFD0", name: "Glacier" },
    { n: "Obsidian", c: "#2E2A38", b: "#4B4560", d: "#FF6A3D", w: "#1C1924", name: "Onyx", ...DARK, eye: "#FF8A3D", ring: false },
  ],
  ghost: [
    { n: "Classic", c: "#F3F5FF", b: "#FFFFFF", d: "#B9C1E6", name: "Boo" },
    { n: "Mint", c: "#C9F7E4", b: "#FFFFFF", d: "#86D9B6", name: "Wisp" },
    { n: "Lavender", c: "#E3D9FF", b: "#FFFFFF", d: "#A996E6", name: "Mist" },
    { n: "Pumpkin", c: "#FFC58A", b: "#FFE6CC", d: "#E08A3A", name: "Gourd" },
  ],
  robot: [
    { n: "Silver", c: "#BAC6D8", b: "#E2E9F3", d: "#1E2432", eye: "#7FE6FF", name: "Bolt" },
    { n: "Arcade", c: "#F2F4F7", b: "#FFFFFF", d: "#26221F", eye: "#FF8A3D", name: "Byte" },
    { n: "Gold", c: "#E8C46A", b: "#FFF0C2", d: "#2A2418", eye: "#FFE680", name: "Brass" },
    { n: "Retro", c: "#9FC7A2", b: "#D6EDD8", d: "#16261A", eye: "#B6FF8A", name: "Pixel" },
  ],
  frog: [
    { n: "Leaf", c: "#8ED472", b: "#D8F5C8", d: "#4E9A3A", name: "Hops" },
    { n: "Dart", c: "#4DA3FF", b: "#BFE0FF", d: "#1C3C66", pat: "spots", name: "Blue" },
    { n: "Lemon", c: "#FFD84D", b: "#FFF3B8", d: "#C9A21F", name: "Zest" },
    { n: "Strawberry", c: "#FF7A7A", b: "#FFD0D0", d: "#1D1A22", pat: "spots", name: "Berry" },
  ],
  duck: [
    { n: "Duckling", c: "#FFDA55", b: "#FFF2B8", d: "#FF9A3C", name: "Waddles" },
    { n: "White", c: "#FAFAFA", b: "#FFFFFF", d: "#FF9A3C", name: "Puddle" },
    { n: "Mallard", c: "#A38F76", b: "#D8C8B2", d: "#F2C230", pat: "mallard", name: "Drake" },
    { n: "Rubber", c: "#FFE14D", b: "#FFF3A0", d: "#FF6A2F", pat: "gloss", name: "Squeaky" },
  ],
  capybara: [
    { n: "Classic", c: "#BE8E62", b: "#D9B48D", d: "#7C5436", name: "Chill" },
    { n: "With yuzu", c: "#BE8E62", b: "#D9B48D", d: "#7C5436", pat: "yuzu", name: "Yuzu" },
    { n: "Golden", c: "#D9A86A", b: "#EDCB9C", d: "#94683A", name: "Sunny" },
    { n: "Cocoa", c: "#8C6546", b: "#B58E6C", d: "#553A26", name: "Mocha" },
  ],
  axolotl: [
    { n: "Pink", c: "#FFB6D0", b: "#FFE1EC", d: "#FF7FAE", name: "Lotl" },
    { n: "Leucistic", c: "#FFF2F6", b: "#FFFFFF", d: "#FF6F9C", name: "Pearl" },
    { n: "Gold", c: "#FFD37A", b: "#FFF0C8", d: "#FF9F45", name: "Nugget" },
    { n: "Blue", c: "#9FC9FF", b: "#DDEBFF", d: "#6C9BFF", name: "Splash" },
    { n: "Melanoid", c: "#3D3A4A", b: "#56526A", d: "#7A6C9A", name: "Inky", ...DARK },
  ],
};
export const petDef = (k: PetKind) => PETS.find((p) => p[0] === k) || PETS[0];
export const petLook = (k: PetKind, i = 0): PetLook => { const L = PET_LOOKS[k] || PET_LOOKS.cat; return L[((i % L.length) + L.length) % L.length]; };
export const petName = (p: { k: PetKind; look: number; name: string }) => (p.name || "").trim() || petLook(p.k, p.look).name;
let PET_UID = 0;
export function petSvg(k: PetKind, acc: PetGear = "none", look = 0): string {
  const top = petDef(k)[2]; const L = petLook(k, look); const { c, b, d } = L;
  const ink = L.eye || "#1D1A22", line = L.line || "#1D1A22";
  const BODY = "M50 38C73 38 85 53 85 70C85 86 71 93 50 93C29 93 15 86 15 70C15 53 27 38 50 38Z";
  let shape = BODY, back = "", body = "", front = "", eyeY = 63, blush = true;
  let eyes: string | null = null;
  let mouth = `<path d="M45.5 71.5q2.2 2.6 4.5 0q2.3 2.6 4.5 0" stroke="${line}" stroke-width="2.2" fill="none" stroke-linecap="round"/>`;
  switch (k) {
    case "cat":
      back = `<path d="M84 82q16-6 8-26" stroke="${c}" stroke-width="8" fill="none" stroke-linecap="round"/><path d="M24 52L27 25L45 41Z" fill="${c}"/><path d="M76 52L73 25L55 41Z" fill="${c}"/><path d="M28 45L29 32L38 41Z" fill="${L.pat === "points" ? d : "#F7A8B8"}"/><path d="M72 45L71 32L62 41Z" fill="${L.pat === "points" ? d : "#F7A8B8"}"/>`;
      front = `<path d="M22 70h10M23 75l9-2M78 70H68M77 75l-9-2" stroke="${L.ring ? "#fff" : d}" stroke-width="1.6" stroke-linecap="round" opacity=".6"/>`;
      break;
    case "dog":
      front = `<path d="M24 45q-13 10-7 29q9 3 13-9q2-12-6-20Z" fill="${d}"/><path d="M76 45q13 10 7 29q-9 3-13-9q-2-12 6-20Z" fill="${d}"/>${L.pat ? "" : `<ellipse cx="61" cy="61" rx="8" ry="7" fill="${b}" opacity=".9"/>`}<ellipse cx="50" cy="69" rx="3.6" ry="2.6" fill="#1D1A22"/>`;
      mouth = `<path d="M50 71.5v2M46 74q4 3 8 0" stroke="${L.pat || !L.ring ? "#1D1A22" : line}" stroke-width="2" fill="none" stroke-linecap="round"/>`;
      break;
    case "fox":
      back = `<ellipse cx="84" cy="74" rx="11" ry="20" transform="rotate(35 84 74)" fill="${c}"/><ellipse cx="91" cy="62" rx="6" ry="7" transform="rotate(35 91 62)" fill="${b}"/><path d="M22 54L26 24L46 42Z" fill="${c}"/><path d="M78 54L74 24L54 42Z" fill="${c}"/><path d="M24 36L26 24L33 31Z" fill="${d}"/><path d="M76 36L74 24L67 31Z" fill="${d}"/>`;
      front = `<path d="M22 70Q34 88 50 88Q66 88 78 70Q66 78 50 74Q34 78 22 70Z" fill="${b}"/><ellipse cx="50" cy="69" rx="3" ry="2.2" fill="#1D1A22"/>`;
      mouth = `<path d="M46 73q4 3 8 0" stroke="#1D1A22" stroke-width="2" fill="none" stroke-linecap="round"/>`;
      break;
    case "bunny":
      back = `<ellipse cx="38" cy="24" rx="7.5" ry="20" transform="rotate(-10 38 24)" fill="${c}"/><ellipse cx="62" cy="24" rx="7.5" ry="20" transform="rotate(10 62 24)" fill="${c}"/><ellipse cx="38" cy="26" rx="3.6" ry="14" transform="rotate(-10 38 26)" fill="${d}"/><ellipse cx="62" cy="26" rx="3.6" ry="14" transform="rotate(10 62 26)" fill="${d}"/>`;
      front = `<ellipse cx="50" cy="69" rx="2.8" ry="2" fill="#F48FAE"/><rect x="47.6" y="73" width="4.8" height="4" rx="1" fill="#fff" stroke="#1D1A22" stroke-width="1"/>`;
      mouth = `<path d="M50 71v2M46.5 73q3.5 2 7 0" stroke="${L.pat === "dutch" ? "#1D1A22" : line}" stroke-width="1.8" fill="none" stroke-linecap="round"/>`;
      break;
    case "slime":
      shape = "M50 34C74 34 88 56 88 77C88 89 80 93 50 93C20 93 12 89 12 77C12 56 26 34 50 34Z";
      body = `<path d="${shape}" fill="${c}" fill-opacity=".94"/><circle cx="24" cy="92" r="4" fill="${c}"/><circle cx="74" cy="93" r="3" fill="${c}"/>`;
      front = `<ellipse cx="33" cy="50" rx="8" ry="5" transform="rotate(-30 33 50)" fill="#fff" opacity=".5"/><circle cx="43" cy="44" r="2" fill="#fff" opacity=".55"/>`;
      break;
    case "dragon":
      back = `<path d="M20 62q-16-12-12-28q8 10 20 12Z" fill="${L.w}"/><path d="M80 62q16-12 12-28q-8 10-20 12Z" fill="${L.w}"/><path d="M34 43l-5-15l12 10Z" fill="${d}"/><path d="M66 43l5-15l-12 10Z" fill="${d}"/><path d="M84 84q12 2 12-10l-6 4" stroke="${c}" stroke-width="6" fill="none" stroke-linecap="round"/>`;
      front = `<ellipse cx="50" cy="80" rx="18" ry="11" fill="${b}"/><path d="M44 38l3-6l3 6M50 37l3-6l3 6" fill="${d}"/>`;
      break;
    case "ghost":
      shape = "M50 30C73 30 84 48 84 66V92Q78.3 86 72.7 92Q67 98 61.3 92Q55.7 86 50 92Q44.3 98 38.7 92Q33 86 27.3 92Q21.7 98 16 92V66C16 48 27 30 50 30Z";
      body = `<path d="${shape}" fill="${c}" fill-opacity=".95"/>`;
      mouth = `<ellipse cx="50" cy="73" rx="3" ry="3.6" fill="#1D1A22"/>`; eyeY = 60;
      break;
    case "robot":
      back = `<path d="M50 38V25" stroke="#8A97AC" stroke-width="3"/><circle class="bulb" cx="50" cy="23" r="5"/>`;
      shape = "M35 38H65A19 19 0 0 1 84 57V74A19 19 0 0 1 65 93H35A19 19 0 0 1 16 74V57A19 19 0 0 1 35 38Z";
      body = `<path d="${shape}" fill="${c}"/><rect x="11" y="58" width="7" height="16" rx="3.5" fill="#8A97AC"/><rect x="82" y="58" width="7" height="16" rx="3.5" fill="#8A97AC"/><rect x="26" y="49" width="48" height="31" rx="11" fill="${d}"/>`;
      eyes = `<g class="eyes"><rect x="35" y="58" width="8" height="9" rx="3" fill="${ink}"/><rect x="57" y="58" width="8" height="9" rx="3" fill="${ink}"/></g><g class="eyes-c"><path d="M35 63h8M57 63h8" stroke="${ink}" stroke-width="2.4" stroke-linecap="round"/></g>`;
      mouth = `<path d="M45 72q5 3 10 0" stroke="${ink}" stroke-width="2" fill="none" stroke-linecap="round"/>`; blush = false;
      break;
    case "frog":
      back = `<circle cx="35" cy="45" r="12" fill="${c}"/><circle cx="65" cy="45" r="12" fill="${c}"/>`;
      front = `<ellipse cx="50" cy="82" rx="20" ry="9" fill="${b}"/>`;
      eyeY = 45; mouth = `<path d="M34 68Q50 80 66 68" stroke="#1D1A22" stroke-width="2.4" fill="none" stroke-linecap="round"/>`;
      break;
    case "duck":
      back = `<path d="M50 39q-5-11 2-13q-2 6 4 9" fill="${L.pat === "mallard" ? "#2E8B57" : c}"/><ellipse cx="17" cy="72" rx="6" ry="11" transform="rotate(20 17 72)" fill="${L.pat === "mallard" ? "#7D6A55" : "#F4C93E"}"/><ellipse cx="83" cy="72" rx="6" ry="11" transform="rotate(-20 83 72)" fill="${L.pat === "mallard" ? "#7D6A55" : "#F4C93E"}"/>`;
      front = `<ellipse cx="50" cy="71.5" rx="10" ry="4.8" fill="${d}"/><path d="M41 71.5h18" stroke="#00000033" stroke-width="1.4"/>`;
      mouth = "";
      break;
    case "capybara":
      back = `<circle cx="29" cy="47" r="5.5" fill="${d}"/><circle cx="71" cy="47" r="5.5" fill="${d}"/>`;
      shape = "M50 44C76 44 88 56 88 72C88 87 73 93 50 93C27 93 12 87 12 72C12 56 24 44 50 44Z";
      front = `<rect x="36" y="64" width="28" height="19" rx="9.5" fill="${b}"/><ellipse cx="45" cy="70" rx="1.8" ry="1.4" fill="${d}"/><ellipse cx="55" cy="70" rx="1.8" ry="1.4" fill="${d}"/>`;
      eyeY = 59; mouth = `<path d="M46 77q4 2 8 0" stroke="${d}" stroke-width="2" fill="none" stroke-linecap="round"/>`;
      eyes = `<g class="eyes"><ellipse cx="36" cy="59" rx="3.6" ry="2.6" fill="#1D1A22"/><ellipse cx="64" cy="59" rx="3.6" ry="2.6" fill="#1D1A22"/><path d="M31.5 57.2h9M59.5 57.2h9" stroke="${c}" stroke-width="2.4"/></g><g class="eyes-c"><path d="M32 60q4 2 8 0M60 60q4 2 8 0" stroke="#1D1A22" stroke-width="2.2" fill="none" stroke-linecap="round"/></g>`;
      break;
    case "axolotl":
      back = `<path d="M24 56l-13-9M22 63H7M24 70l-13 8M76 56l13-9M78 63h15M76 70l13 8" stroke="${d}" stroke-width="5.5" stroke-linecap="round"/>`;
      front = `<ellipse cx="50" cy="82" rx="17" ry="9" fill="${b}"/>`;
      mouth = `<path d="M40 70Q50 78 60 70" stroke="${line}" stroke-width="2.2" fill="none" stroke-linecap="round"/>`;
      break;
  }
  if (!body) body = `<path d="${shape}" fill="${c}"/>`;
  // coat patterns live inside the body outline
  const id = `pc${++PET_UID}`;
  const P = ({
    tabby: `<path d="M46 42v6M50 41v7M54 42v6M15 64h8M15 71h8M85 64h-8M85 71h-8" stroke="${d}" stroke-width="2.6" stroke-linecap="round"/>`,
    tux: `<ellipse cx="50" cy="90" rx="19" ry="12" fill="${b}"/><ellipse cx="50" cy="73" rx="11" ry="7.5" fill="${b}"/>`,
    calico: `<ellipse cx="29" cy="50" rx="15" ry="11" fill="#F2A65A"/><ellipse cx="74" cy="82" rx="14" ry="10" fill="${d}"/><ellipse cx="70" cy="44" rx="8" ry="6" fill="${d}"/>`,
    points: `<ellipse cx="50" cy="71" rx="12" ry="9" fill="${d}" opacity=".5"/>`,
    spots: `<g fill="${d}"><circle cx="30" cy="54" r="3.6"/><circle cx="70" cy="50" r="3"/><circle cx="24" cy="78" r="3.2"/><circle cx="76" cy="78" r="3.8"/><circle cx="58" cy="88" r="2.6"/><circle cx="40" cy="88" r="2.6"/></g>`,
    mask: `<path d="M20 76Q28 60 40 67Q50 61 60 67Q72 60 80 76Q70 94 50 94Q30 94 20 76Z" fill="${b}"/><circle cx="39" cy="54" r="3" fill="${b}"/><circle cx="61" cy="54" r="3" fill="${b}"/>`,
    blaze: `<rect x="46" y="36" width="8" height="30" rx="4" fill="${b}"/><ellipse cx="50" cy="75" rx="14" ry="10" fill="${b}"/>`,
    dutch: `<rect x="0" y="72" width="100" height="30" fill="${b}"/><path d="M44 38L50 72L56 38Z" fill="${b}"/>`,
    mallard: `<ellipse cx="50" cy="46" rx="42" ry="17" fill="#2E8B57"/><path d="M22 61Q50 67 78 61" stroke="#fff" stroke-width="3" fill="none"/>`,
    gloss: `<ellipse cx="32" cy="52" rx="9" ry="5" transform="rotate(-30 32 52)" fill="#fff" opacity=".6"/>`,
  } as Record<string, string>)[L.pat ?? ""] || "";
  const coat = P ? `<clipPath id="${id}"><path d="${shape}"/></clipPath><g clip-path="url(#${id})">${P}</g>` : "";
  const yuzu = L.pat === "yuzu" ? `<g><circle cx="58" cy="${top - 4}" r="8" fill="#FFB23F"/><circle cx="55.5" cy="${top - 6.5}" r="2" fill="#fff" opacity=".55"/><path d="M58 ${top - 12}q5-6 10-2q-5 4-10 2Z" fill="#6DBE45"/></g>` : "";
  const ex = eyeY === 45 ? [35, 65] : [39, 61];
  const ring = L.ring ? ` stroke="#fff" stroke-opacity=".9" stroke-width="1.5"` : "";
  eyes = eyes || `<g class="eyes"><ellipse cx="${ex[0]}" cy="${eyeY}" rx="4.3" ry="5.1" fill="${ink}"${ring}/><ellipse cx="${ex[1]}" cy="${eyeY}" rx="4.3" ry="5.1" fill="${ink}"${ring}/><circle cx="${ex[0] + 1.5}" cy="${eyeY - 1.9}" r="1.6" fill="#fff"/><circle cx="${ex[1] + 1.5}" cy="${eyeY - 1.9}" r="1.6" fill="#fff"/></g>
    <g class="eyes-c"><path d="M${ex[0] - 4} ${eyeY + 1}q4 3 8 0M${ex[1] - 4} ${eyeY + 1}q4 3 8 0" stroke="${L.ring || L.eye && L.eye !== "#1D1A22" ? "#fff" : "#1D1A22"}" stroke-width="2.4" fill="none" stroke-linecap="round"/></g>`;
  const bl = blush ? `<ellipse cx="29" cy="${eyeY + 8}" rx="5" ry="3" fill="#FF8FA3" opacity=".45"/><ellipse cx="71" cy="${eyeY + 8}" rx="5" ry="3" fill="#FF8FA3" opacity=".45"/>` : "";
  // dark coats get a soft light rim so they don't vanish into the dark UI
  const lum = ((h: string) => { const n = parseInt(h.slice(1), 16); return (.2126 * (n >> 16) + .7152 * (n >> 8 & 255) + .0722 * (n & 255)) / 255; })(c);
  return `<svg class="pet-svg ${lum < .32 ? "dark-coat" : ""}" viewBox="0 0 100 100" aria-hidden="true">${back}${body}${coat}${front}${bl}${eyes}${mouth}${yuzu}${petAcc(acc, top)}</svg>`;
}
function petAcc(a: PetGear, top: number) {
  const dy = top - 38;
  if (a === "bandana") return `<path d="M24 82Q50 91 76 82L74 88Q50 97 26 88Z" fill="#FF5A5F"/><path d="M60 88l9 7l2-9Z" fill="#E0474C"/>`;
  if (a === "headset") return `<g transform="translate(0 ${dy})"><path d="M20 58Q20 30 50 30Q80 30 80 58" stroke="#2A2A31" stroke-width="5" fill="none"/><rect x="12" y="52" width="11" height="18" rx="5.5" fill="#2A2A31"/><rect x="77" y="52" width="11" height="18" rx="5.5" fill="#2A2A31"/><rect x="14" y="55" width="7" height="12" rx="3.5" class="bulb"/><path d="M17 69q2 10 18 9" stroke="#2A2A31" stroke-width="2.6" fill="none" stroke-linecap="round"/><circle cx="36" cy="78" r="2.6" fill="#2A2A31"/></g>`;
  if (a === "cap") return `<g transform="translate(0 ${dy})"><path d="M27 45Q30 24 50 24Q70 24 73 45Z" fill="#4C8DFF"/><path d="M62 43Q82 38 90 45Q78 50 62 46Z" fill="#3A74E0"/><circle cx="50" cy="24.5" r="2.4" fill="#3A74E0"/></g>`;
  if (a === "crown") return `<g transform="translate(0 ${dy})"><path d="M35 40L37 22L44 31L50 18L56 31L63 22L65 40Z" fill="#FFC83D" stroke="#E0A21A" stroke-width="1.4" stroke-linejoin="round"/><circle cx="50" cy="33" r="2.4" fill="#FF5A5F"/><circle cx="42" cy="36" r="1.7" fill="#4C8DFF"/><circle cx="58" cy="36" r="1.7" fill="#4C8DFF"/></g>`;
  return "";
}
export const heartSvg = (c = "#FF6B8B") => `<svg viewBox="0 0 24 24"><path d="M12 21s-7.5-4.6-9.7-9.3C.8 8.4 2.7 4.5 6.4 4.5c2.2 0 3.6 1.2 4.6 2.7 1-1.5 2.4-2.7 4.6-2.7 3.7 0 5.6 3.9 4.1 7.2C19.5 16.4 12 21 12 21Z" fill="${c}"/></svg>`;

/* ---------- lines, in the hype-friend voice ---------- */
export const PET_LINES: Record<"keep" | "wild" | "scan" | "pet" | "wake" | "roam", string[]> = {
  keep: ["Into the pile!", "Banger.", "Chat's gonna love that", "Certified keeper", "Yes. That one."],
  wild: ["THAT ONE THOUGH", "Clip of the week?", "Chat was screaming", "Okay that's elite"],
  scan: ["Scan's done!", "Fresh moments, come look", "Deck's ready for you"],
  pet: ["Hi!", "Hehe", "More clips?", "We're so back", "Just vibing", "You're doing great"],
  wake: ["Oh! You're back", "Was just resting my eyes"],
  roam: ["Stretching my legs", "Watching the scan", "Ooh, a new VOD"],
};
export const petPick = (a: string[]) => a[Math.floor(Math.random() * a.length)];

/** Swatch for a look: coat color with its accent. */
export const lookSwatch = (L: PetLook) =>
  `linear-gradient(135deg, ${L.c} 0 55%, ${L.pat === "tux" || L.pat === "dutch" || L.pat === "mask" || L.pat === "blaze" ? L.b : L.d} 55% 100%)`;
