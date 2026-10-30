// The wind speed colour map: viridis, 256 entries as six-digit hex (generated from seaborn's table - do not edit by
// hand), so the page needs no colour library and colours are computed synchronously.

const VIRIDIS_HEX =
  "44015444025645045745055946075a46085c460a5d460b5e470d60470e61471063471164471365481467481668481769" +
  "48186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a" +
  "472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f85" +
  "4240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b" +
  "3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d" +
  "33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e" +
  "2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e" +
  "26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d" +
  "21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f88" +
  "1fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad81" +
  "28ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc74" +
  "3fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc863" +
  "5ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d" +
  "84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32" +
  "addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21a" +
  "d8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725";

let viridis = null;

/** The wind colour at position t (0..1, clamped): [r, g, b] in 0-255, interpolated between the table's entries. */
export function windColour(t) {
  if (!viridis) {
    viridis = new Uint8Array(256 * 3);
    for (let i = 0; i < 256 * 3; i++) viridis[i] = parseInt(VIRIDIS_HEX.substr(i * 2, 2), 16);
  }
  const x = Math.min(1, Math.max(0, t)) * 255;
  const i = Math.floor(x);
  const j = Math.min(255, i + 1);
  const f = x - i;
  return [
    viridis[i * 3] * (1 - f) + viridis[j * 3] * f,
    viridis[i * 3 + 1] * (1 - f) + viridis[j * 3 + 1] * f,
    viridis[i * 3 + 2] * (1 - f) + viridis[j * 3 + 2] * f,
  ];
}
