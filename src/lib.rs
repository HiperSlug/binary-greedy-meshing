use std::collections::HashSet;

use bit_iter::BitIter;
use enum_map::{EnumMap, enum_map};

mod types;

use types::Face::*;
pub use types::*;

const BITS: u32 = 6;

pub const LEN: usize = 1 << BITS;
pub const SQUARE: usize = LEN * LEN;
pub const CUBE: usize = LEN * LEN * LEN;

const SHIFT_X_3D: u32 = 0 * BITS;
const SHIFT_Y_3D: u32 = 1 * BITS;
const SHIFT_Z_3D: u32 = 2 * BITS;

const STRIDE_X_3D: usize = 1 << SHIFT_X_3D;
const STRIDE_Y_3D: usize = 1 << SHIFT_Y_3D;
const STRIDE_Z_3D: usize = 1 << SHIFT_Z_3D;

const SHIFT_Y_2D: u32 = 0 * BITS;
const SHIFT_Z_2D: u32 = 1 * BITS;

const STRIDE_Y_2D: usize = 1 << SHIFT_Y_2D;
const STRIDE_Z_2D: usize = 1 << SHIFT_Z_2D;

const UPWARD_STRIDE_X: usize = STRIDE_X_3D;

const FORWARD_STRIDE_X: usize = STRIDE_X_3D;
const FORWARD_STRIDE_Y: usize = STRIDE_Y_3D;

const PAD_MASK: u64 = (1 << 63) | 1;

#[inline(always)]
fn linearize_2d(y: usize, z: usize) -> usize {
    (y << SHIFT_Y_2D) | (z << SHIFT_Z_2D)
}

#[inline(always)]
pub fn linearize_3d(x: usize, y: usize, z: usize) -> usize {
    (x << SHIFT_X_3D) | (y << SHIFT_Y_3D) | (z << SHIFT_Z_3D)
}

#[inline(always)]
fn linearize_2d_to_3d(x: usize, i_2d: usize) -> usize {
    (x << SHIFT_X_3D) | (i_2d << SHIFT_Y_3D)
}

#[inline(always)]
fn offset_3d(face: Face) -> isize {
    match face {
        PosX => STRIDE_X_3D as isize,
        NegX => -(STRIDE_X_3D as isize),
        PosY => STRIDE_Y_3D as isize,
        NegY => -(STRIDE_Y_3D as isize),
        PosZ => STRIDE_Z_3D as isize,
        NegZ => -(STRIDE_Z_3D as isize),
    }
}

#[inline(always)]
fn adj_opaque(face: Face, pad_opaque: u64, opaque_masks: &[u64; SQUARE], i_2d: usize) -> u64 {
    match face {
        PosX => pad_opaque >> 1,
        NegX => pad_opaque << 1,
        PosY => opaque_masks[i_2d + STRIDE_Y_2D],
        NegY => opaque_masks[i_2d - STRIDE_Y_2D],
        PosZ => opaque_masks[i_2d + STRIDE_Z_2D],
        NegZ => opaque_masks[i_2d - STRIDE_Z_2D],
    }
}

/// Reusable buffers for meshing
pub struct Mesher {
    // divided into two structures so I can pass `&mut self.scratch` as an argument in function calls
    scratch: Vec<Quad>,
    inner: InnerMesher,
}

impl Default for Mesher {
    fn default() -> Self {
        Self {
            scratch: Vec::new(),
            inner: InnerMesher {
                visible_masks: Box::new(enum_map! { _ => [0; SQUARE] }),
                forward_merged: Box::new([0; SQUARE]),
                upward_merged: Box::new([0; LEN]),
            },
        }
    }
}

struct InnerMesher {
    visible_masks: Box<EnumMap<Face, [u64; SQUARE]>>,
    forward_merged: Box<[u8; SQUARE]>,
    upward_merged: Box<[u8; LEN]>,
}

impl InnerMesher {
    fn build_visible_slow(
        &mut self,
        voxels: &[u16; CUBE],
        transparents: &HashSet<u16>,
        xs: impl Iterator<Item = usize> + Clone,
        ys: impl Iterator<Item = usize> + Clone,
        zs: impl Iterator<Item = usize>,
    ) {
        for array in self.visible_masks.values_mut() {
            array.fill(0);
        }

        for z in zs {
            for y in ys.clone() {
                let i_2d = linearize_2d(y, z);

                for x in xs.clone() {
                    let i_3d = linearize_3d(x, y, z);
                    let voxel = voxels[i_3d];

                    if voxel == 0 {
                        continue;
                    }

                    let bit = 1 << x;

                    for face in Face::ALL {
                        let offset_3d = offset_3d(face);

                        let adj_i_3d = i_3d.wrapping_add_signed(offset_3d);
                        let adj_voxel = voxels[adj_i_3d];

                        if adj_voxel == 0
                            || (voxel != adj_voxel && transparents.contains(&adj_voxel))
                        {
                            self.visible_masks[face][i_2d] |= bit;
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn build_all_visible_slow(&mut self, voxels: &[u16; CUBE], transparents: &HashSet<u16>) {
        self.build_visible_slow(voxels, transparents, 1..LEN - 1, 1..LEN - 1, 1..LEN - 1);
    }

    fn fast_row_handler(
        &mut self,
        voxels: &[u16; CUBE],
        opaque_masks: &[u64; SQUARE],
        transparent_masks: &[u64; SQUARE],
        xs: u64,
        y: usize,
        z: usize,
    ) {
        let i_2d = linearize_2d(y, z);

        let pad_opaque = opaque_masks[i_2d];
        let opaque = pad_opaque & !PAD_MASK & xs;

        let transparent = transparent_masks[i_2d] & !PAD_MASK & xs;

        if opaque == 0 && transparent == 0 {
            for visible_masks in self.visible_masks.values_mut() {
                visible_masks[i_2d] = 0;
            }
            return;
        }

        for (face, visible_masks) in self.visible_masks.iter_mut() {
            let offset_3d = offset_3d(face);

            let adj_opaque = adj_opaque(face, pad_opaque, opaque_masks, i_2d);

            visible_masks[i_2d] = opaque & !adj_opaque;

            for x in BitIter::from(transparent & !adj_opaque) {
                let i_3d = linearize_2d_to_3d(x, i_2d);
                let voxel = voxels[i_3d];

                let adj_i_3d = i_3d.wrapping_add_signed(offset_3d);
                let adj_voxel = voxels[adj_i_3d];

                if voxel != adj_voxel {
                    visible_masks[i_2d] |= 1 << x;
                }
            }
        }
    }

    fn build_visible(
        &mut self,
        voxels: &[u16; CUBE],
        opaque_masks: &[u64; SQUARE],
        transparent_masks: &[u64; SQUARE],
        xs: u64,
        ys: u64,
        zs: u64,
    ) {
        let inv_ys = !ys & !PAD_MASK;
        let inv_zs = !zs & !PAD_MASK;

        for z in BitIter::from(zs) {
            for y in 1..LEN - 1 {
                self.fast_row_handler(voxels, opaque_masks, transparent_masks, !0, y, z);
            }
        }

        for z in BitIter::from(inv_zs) {
            for y in BitIter::from(ys) {
                self.fast_row_handler(voxels, opaque_masks, transparent_masks, !0, y, z);
            }
        }

        for z in BitIter::from(inv_zs) {
            for y in BitIter::from(inv_ys) {
                self.fast_row_handler(voxels, opaque_masks, transparent_masks, xs, y, z);
            }
        }
    }

    fn build_all_visible(
        &mut self,
        voxels: &[u16; CUBE],
        opaque_masks: &[u64; SQUARE],
        transparent_masks: &[u64; SQUARE],
    ) {
        for z in 1..LEN - 1 {
            for y in 1..LEN - 1 {
                self.fast_row_handler(voxels, opaque_masks, transparent_masks, !0, y, z);
            }
        }
    }

    fn face_merging(&mut self, voxels: &[u16; CUBE]) -> EnumMap<Face, Vec<Quad>> {
        let mut map = EnumMap::default();

        for (face, output) in &mut map {
            match face {
                PosX | NegX => self.merge_x(voxels, !0, face, output),
                PosY | NegY => self.merge_y(voxels, 1..LEN - 1, face, output),
                PosZ | NegZ => self.merge_z(voxels, 1..LEN - 1, face, output),
            }
        }

        map
    }

    fn merge_x(&mut self, voxels: &[u16; CUBE], xs: u64, face: Face, output: &mut Vec<Quad>) {
        for z in 1..LEN - 1 {
            for y in 1..LEN - 1 {
                let i_2d = linearize_2d(y, z);

                let visible = self.visible_masks[face][i_2d] & xs;
                let upward_visible = self.visible_masks[face][i_2d + STRIDE_Y_2D] & xs;
                let forward_visible = self.visible_masks[face][i_2d + STRIDE_Z_2D] & xs;

                for x in BitIter::from(visible) {
                    let upward_i = x;
                    let forward_i = linearize_2d(x, y);

                    let i_3d = linearize_2d_to_3d(x, i_2d);
                    let voxel = voxels[i_3d];

                    // forward merging
                    if self.upward_merged[upward_i] == 0
                        && (forward_visible >> x) & 1 != 0
                        && voxel == voxels[i_3d + STRIDE_Z_3D]
                    {
                        self.forward_merged[forward_i] += 1;
                        continue;
                    }

                    // upward merging
                    if (upward_visible >> x) & 1 != 0
                        && self.forward_merged[forward_i]
                            == self.forward_merged[forward_i + FORWARD_STRIDE_Y]
                        && voxel == voxels[i_3d + STRIDE_Y_3D]
                    {
                        self.forward_merged[forward_i] = 0;
                        self.upward_merged[upward_i] += 1;
                        continue;
                    }

                    // finish
                    output.push({
                        let forward_merged = self.forward_merged[forward_i] as u32;
                        let upward_merged = self.upward_merged[upward_i] as u32;

                        let x = x as u32;
                        let y = y as u32 - upward_merged;
                        let z = z as u32 - forward_merged;

                        let w = forward_merged + 1;
                        let h = upward_merged + 1;

                        let id = voxel as u32;

                        Quad::new(x, y, z, w, h, id)
                    });

                    self.forward_merged[forward_i] = 0;
                    self.upward_merged[upward_i] = 0;
                }
            }
        }

        output.sort_unstable_by_key(|q| q.x());
    }

    fn merge_y(
        &mut self,
        voxels: &[u16; CUBE],
        ys: impl Iterator<Item = usize> + Clone,
        face: Face,
        output: &mut Vec<Quad>,
    ) {
        for z in 1..LEN - 1 {
            for y in ys.clone() {
                let i_2d = linearize_2d(y, z);

                let mut visible = self.visible_masks[face][i_2d];
                let forward_visible = self.visible_masks[face][i_2d + STRIDE_Z_2D];

                while visible != 0 {
                    let x = visible.trailing_zeros() as usize;

                    let forward_i = linearize_2d(x, y);

                    let i_3d = linearize_2d_to_3d(x, i_2d);
                    let voxel = voxels[i_3d];

                    // forward merging
                    if (forward_visible >> x) & 1 != 0 && voxel == voxels[i_3d + STRIDE_Z_3D] {
                        self.forward_merged[forward_i] += 1;
                        visible &= visible - 1;
                        continue;
                    }

                    // rightward merging
                    let mut next_x = x + 1;
                    let mut next_forward_i = forward_i + FORWARD_STRIDE_X;
                    let mut next_i_3d = i_3d + STRIDE_X_3D;

                    while next_x < LEN - 1
                        && (visible >> next_x) & 1 != 0
                        && self.forward_merged[forward_i] == self.forward_merged[next_forward_i]
                        && voxel == voxels[next_i_3d]
                    {
                        self.forward_merged[next_forward_i] = 0;

                        next_x += 1;
                        next_forward_i += FORWARD_STRIDE_X;
                        next_i_3d += STRIDE_X_3D;
                    }

                    let right_merged = next_x - x;
                    visible &= !((1 << next_x) - 1);

                    // finish
                    output.push({
                        let forward_merged = self.forward_merged[forward_i] as u32;

                        let x = x as u32;
                        let y = y as u32;
                        let z = z as u32 - forward_merged;

                        let w = right_merged as u32;
                        let h = forward_merged + 1;

                        let id = voxel as u32;

                        Quad::new(x, y, z, w, h, id)
                    });

                    self.forward_merged[forward_i] = 0
                }
            }
        }

        output.sort_unstable_by_key(|q| q.y());
    }

    fn merge_z(
        &mut self,
        voxels: &[u16; CUBE],
        zs: impl Iterator<Item = usize>,
        face: Face,
        output: &mut Vec<Quad>,
    ) {
        for z in zs {
            for y in 1..LEN - 1 {
                let i_2d = linearize_2d(y, z);

                let mut visible = self.visible_masks[face][i_2d];
                let upward_visible = self.visible_masks[face][i_2d + STRIDE_Y_2D];

                while visible != 0 {
                    let x = visible.trailing_zeros() as usize;

                    let upward_i = x as usize;

                    let i_3d = linearize_2d_to_3d(x, i_2d);
                    let voxel = voxels[i_3d];

                    // upward merging
                    if (upward_visible >> x) & 1 != 0 && voxel == voxels[i_3d + STRIDE_Y_3D] {
                        self.upward_merged[upward_i] += 1;
                        visible &= visible - 1;
                        continue;
                    }

                    // rightward merging
                    let mut next_x = x + 1;
                    let mut next_upward_i = upward_i + FORWARD_STRIDE_X;
                    let mut next_i_3d = i_3d + STRIDE_X_3D;

                    while next_x < LEN - 1
                        && (visible >> next_x) & 1 != 0
                        && self.upward_merged[upward_i] == self.upward_merged[next_upward_i]
                        && voxel == voxels[next_i_3d]
                    {
                        self.upward_merged[next_upward_i] = 0;

                        next_x += 1;
                        next_upward_i += UPWARD_STRIDE_X;
                        next_i_3d += STRIDE_X_3D;
                    }

                    let right_merged = next_x - x;
                    visible &= !((1 << next_x) - 1);

                    // finish
                    output.push({
                        let upward_merged = self.upward_merged[upward_i] as u32;

                        let x = x as u32;
                        let y = y as u32 - upward_merged;
                        let z = z as u32;

                        let w = right_merged as u32;
                        let h = upward_merged + 1;

                        let id = voxel as u32;

                        Quad::new(x, y, z, w, h, id)
                    });

                    self.upward_merged[upward_i] = 0;
                }
            }
        }
    }
}

impl Mesher {
    pub fn new() -> Self {
        Self::default()
    }

    /// Meshes a voxel buffer representing a chunk, using an opaque and transparent mask with 1 u64 per column with 1 bit per voxel in the column,
    /// signaling if the voxel is opaque or transparent.
    /// This is ~4x faster than the regular mesh method but requires maintaining 2 masks for each chunk.
    /// See https://github.com/Inspirateur/binary-greedy-meshing?tab=readme-ov-file#what-to-do-with-mesh_dataquads for using the output
    pub fn mesh(
        &mut self,
        voxels: &[u16; CUBE],
        opaque_masks: &[u64; SQUARE],
        transparent_masks: &[u64; SQUARE],
    ) -> QuadMesh {
        self.inner
            .build_all_visible(voxels, opaque_masks, transparent_masks);
        QuadMesh(self.inner.face_merging(voxels))
    }

    /// Meshes a voxel buffer representing a chunk, using a BTreeSet signaling which voxel values are transparent.
    /// This is ~4x slower than the fast_mesh method but does not require maintaining 2 masks for each chunk.
    /// See https://github.com/Inspirateur/binary-greedy-meshing?tab=readme-ov-file#what-to-do-with-mesh_dataquads for using the output
    pub fn slow_mesh(&mut self, voxels: &[u16; CUBE], transparents: &HashSet<u16>) -> QuadMesh {
        self.inner.build_all_visible_slow(voxels, transparents);
        QuadMesh(self.inner.face_merging(voxels))
    }

    pub fn remesh_slow(
        &mut self,
        voxels: &[u16; CUBE],
        transparents: &HashSet<u16>,
        mesh: &mut QuadMesh,
        changes: MeshChanges,
    ) {
        let [xs, ys, zs] = changes
            .to_array()
            .map(|x| ((x << 1) | (x >> 1) | x) & !PAD_MASK);

        self.inner.build_visible_slow(
            voxels,
            transparents,
            BitIter::from(xs),
            BitIter::from(ys),
            BitIter::from(zs),
        );

        self.merge_and_splice(voxels, mesh, xs, ys, zs);
    }

    pub fn remesh(
        &mut self,
        voxels: &[u16; CUBE],
        opaque_masks: &[u64; SQUARE],
        transparent_masks: &[u64; SQUARE],
        mesh: &mut QuadMesh,
        changes: MeshChanges,
    ) {
        let [xs, ys, zs] = changes
            .to_array()
            .map(|x| ((x << 1) | (x >> 1) | x) & !PAD_MASK);

        self.inner
            .build_visible(voxels, opaque_masks, transparent_masks, xs, ys, zs);

        self.merge_and_splice(voxels, mesh, xs, ys, zs);
    }

    fn merge_and_splice(
        &mut self,
        voxels: &[u16; CUBE],
        mesh: &mut QuadMesh,
        xs: u64,
        ys: u64,
        zs: u64,
    ) {
        fn as_u32(usize: usize) -> u32 {
            usize as u32
        }

        for (face, quads) in &mut mesh.0 {
            self.scratch.clear();
            match face {
                PosX | NegX => {
                    self.inner.merge_x(voxels, xs, face, &mut self.scratch);

                    let mut src_start = 0;
                    for x in BitIter::from(xs).map(as_u32) {
                        let dst_start = quads.partition_point(|q| q.x() < x);
                        let dst_end = quads.partition_point(|q| q.x() <= x);

                        let src_end = self.scratch.partition_point(|q| q.x() <= x);
                        let replace_with = self.scratch[src_start..src_end].iter().copied();
                        src_start = src_end;

                        quads.splice(dst_start..dst_end, replace_with);
                    }
                }
                PosY | NegY => {
                    self.inner
                        .merge_y(voxels, BitIter::from(ys), face, &mut self.scratch);

                    let mut src_start = 0;
                    for y in BitIter::from(ys).map(as_u32) {
                        let dst_start = quads.partition_point(|q| q.y() < y);
                        let dst_end = quads.partition_point(|q| q.y() <= y);

                        let src_end = self.scratch.partition_point(|q| q.y() <= y);
                        let replace_with = self.scratch[src_start..src_end].iter().copied();
                        src_start = src_end;

                        quads.splice(dst_start..dst_end, replace_with);
                    }
                }
                PosZ | NegZ => {
                    self.inner
                        .merge_z(voxels, BitIter::from(zs), face, &mut self.scratch);

                    let mut src_start = 0;
                    for z in BitIter::from(zs).map(as_u32) {
                        let dst_start = quads.partition_point(|q| q.z() < z);
                        let dst_end = quads.partition_point(|q| q.z() <= z);

                        let src_end = self.scratch.partition_point(|q| q.z() <= z);
                        let replace_with = self.scratch[src_start..src_end].iter().copied();
                        src_start = src_end;

                        quads.splice(dst_start..dst_end, replace_with);
                    }
                }
            }
        }
    }
}

/// Compute an opacity mask from a voxel buffer and a HashSet specifying which voxel values are transparent
pub fn compute_opaque_masks(
    voxels: &[u16; CUBE],
    transparents: &HashSet<u16>,
) -> Box<[u64; SQUARE]> {
    let mut opaque_mask = Box::new([0; SQUARE]);
    for z in 0..LEN {
        for y in 0..LEN {
            let i_2d = linearize_2d(y, z);
            for x in 0..LEN {
                let i_3d = linearize_3d(x, y, z);
                let voxel = voxels[i_3d];

                if voxel == 0 || transparents.contains(&voxel) {
                    continue;
                }

                opaque_mask[i_2d] |= 1 << x;
            }
        }
    }
    opaque_mask
}

/// Compute a transparent mask from a voxel buffer and a HashSet specifying which voxel values are transparent
pub fn compute_transparent_masks(
    voxels: &[u16; CUBE],
    transparents: &HashSet<u16>,
) -> Box<[u64; SQUARE]> {
    let mut transparent_mask = Box::new([0; SQUARE]);
    for z in 0..LEN {
        for y in 0..LEN {
            let i_2d = linearize_2d(y, z);
            for x in 0..LEN {
                let i_3d = linearize_3d(x, y, z);
                let voxel = voxels[i_3d];

                if voxel == 0 || !transparents.contains(&voxel) {
                    continue;
                }

                transparent_mask[i_2d] |= 1 << x;
            }
        }
    }
    transparent_mask
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Show quad output on a simple 2 voxels case
    #[test]
    fn test_output() {
        let transparents = HashSet::new();

        let mut voxels = Box::new([0; CUBE]);
        voxels[linearize_3d(1, 1, 1)] = 1;
        voxels[linearize_3d(1, 2, 1)] = 1;

        let opaque_masks = compute_opaque_masks(&voxels, &transparents);
        let transparent_masks = compute_transparent_masks(&voxels, &transparents);

        let mut mesher = Mesher::new();

        let mesh = mesher.mesh(&voxels, &opaque_masks, &transparent_masks);
        for (face, quads) in mesh.0 {
            std::println!("--- Face {face:?} ---\n{quads:?}");
        }
    }

    /// Ensures that mesh and fast_mesh return the same results
    #[test]
    fn same_results() {
        let transparents = HashSet::from([2]);

        let voxels = test_buffer();
        let opaque_masks = compute_opaque_masks(&voxels, &transparents);
        let transparent_masks = compute_transparent_masks(&voxels, &transparents);

        let mut mesher = Mesher::new();

        let mesh = mesher.mesh(&voxels, &opaque_masks, &transparent_masks);
        let slow_mesh = mesher.slow_mesh(&voxels, &transparents);

        assert_eq!(mesh, slow_mesh);
    }

    fn test_buffer() -> Box<[u16; CUBE]> {
        let mut voxels = Box::new([0; CUBE]);
        for x in 1..LEN - 1 {
            for y in 1..LEN - 1 {
                for z in 1..LEN - 1 {
                    let i_3d = linearize_3d(x, y, z);
                    voxels[i_3d] = transparent_sphere(x, y, z);
                }
            }
        }
        voxels
    }

    fn transparent_sphere(x: usize, y: usize, z: usize) -> u16 {
        if x == 8 {
            2
        } else if (x as i32 - 31).pow(2) + (y as i32 - 31).pow(2) + (z as i32 - 31).pow(2)
            < 16 as i32
        {
            1
        } else {
            0
        }
    }
}
