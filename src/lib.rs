mod types;

use enum_map::EnumMap;
use glam::USizeVec3;
use ndshape::{ConstPow2Shape3usize, ConstShape as _, ConstShape2usize};
pub use types::*;

pub const BITS: usize = 6;
pub const LEN: usize = 1 << BITS;
pub const SQUARE: usize = LEN * LEN;
pub const CUBE: usize = LEN * LEN * LEN;
pub type Shape = ConstPow2Shape3usize<BITS, BITS, BITS>;

const STRIDE_X: usize = 1 << Shape::SHIFTS[0];
const STRIDE_Y: usize = 1 << Shape::SHIFTS[1];

const PADDED_LEN: usize = LEN + 1;
type PaddedShape = ConstShape2usize<PADDED_LEN, PADDED_LEN>;
const PADDED_STRIDE_Y_2D: usize = PaddedShape::STRIDES[0];
const PADDED_STRIDE_Z_2D: usize = PaddedShape::STRIDES[1];

#[derive(Debug, Clone)]
pub struct Mesher {
    pub quads: Vec<Quad>,
    /// # Length
    /// Padded along +y and +z (except for the corner) with 0 to avoid branching
    ///
    /// `x` = value \
    /// `0` = padded zeros \
    /// `_` = unused \
    ///
    /// |    |  0 |  1 | 63 | 64 |
    /// |----|----|----|----|----|
    /// |  0 | x  | x  | x  | 0  |
    /// |  1 | x  | x  | x  | 0  |
    /// | 63 | x  | x  | x  | 0  |
    /// | 64 | 0  | 0  | 0  | _  |
    visible_masks: Box<EnumMap<Face, [u64; PaddedShape::SIZE]>>,
    forward_merged: Box<[u8; SQUARE]>,
    upward_merged: Box<[u8; LEN]>,
}

impl Default for Mesher {
    fn default() -> Self {
        Self {
            quads: Vec::new(),
            visible_masks: Box::new(EnumMap::from_array([[0; PaddedShape::SIZE]; 6])),
            forward_merged: Box::new([0; SQUARE]),
            upward_merged: Box::new([0; LEN]),
        }
    }
}

impl Mesher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mesh<X, V, C>(&mut self, view: &V, context: &C) -> EnumMap<Face, usize>
    where
        X: Copy,
        V: MesherView<Voxel = X> + MesherViewAdjacent,
        C: MesherContext<Voxel = X, InnerVoxel: Copy>,
    {
        self.build_visible(view, context);
        let lens = self.face_merging(view, context);
        self.clear_visible();
        lens
    }

    fn clear_visible(&mut self) {
        for mask in self.visible_masks.values_mut() {
            mask.fill(0);
        }
    }

    fn build_visible<X, V, C>(&mut self, view: &V, context: &C)
    where
        V: MesherView<Voxel = X> + MesherViewAdjacent,
        C: MesherContext<Voxel = X, InnerVoxel: Copy>,
    {
        for z in 0..LEN {
            for y in 0..LEN {
                let i_2d = PaddedShape::linearize([y, z]);

                for x in 0..LEN {
                    let pos = USizeVec3::new(x, y, z);

                    let Some(voxel) = context.into_inner(view.get(pos.into())) else {
                        continue;
                    };

                    let bit = 1 << x;

                    for face in Face::ALL {
                        let new_pos = pos
                            .as_uvec3()
                            .wrapping_add_signed(face.to_ivec3())
                            .as_usizevec3();
                        let adj_pos = new_pos % LEN;
                        let external = new_pos != adj_pos;

                        let adj_voxel = if external {
                            match view.get_adjacent(adj_pos.into(), face) {
                                Some(v) => context.into_inner(v),
                                None => continue,
                            }
                        } else {
                            context.into_inner(view.get(adj_pos.into()))
                        };

                        if adj_voxel.map_or(false, |adj_voxel| context.is_visible(voxel, adj_voxel))
                        {
                            self.visible_masks[face][i_2d] |= bit;
                        }
                    }
                }
            }
        }
    }

    fn face_merging<X, V, C>(&mut self, view: &V, context: &C) -> EnumMap<Face, usize>
    where
        X: Copy,
        V: MesherView<Voxel = X>,
        C: MesherContext<Voxel = X>,
    {
        self.quads.clear();

        let mut lens = EnumMap::default();

        for face in Face::ALL {
            match face {
                Face::PosX | Face::NegX => self.merge_x(view, context, face),
                Face::NegY | Face::PosY => self.merge_y(view, context, face),
                Face::PosZ | Face::NegZ => self.merge_z(view, context, face),
            }
            lens[face] = self.quads.len()
        }

        lens
    }

    fn merge_x<X, V, C>(&mut self, view: &V, context: &C, face: Face)
    where
        X: Copy,
        V: MesherView<Voxel = X>,
        C: MesherContext<Voxel = X>,
    {
        for z in 0..LEN {
            for y in 0..LEN {
                let i_2d = PaddedShape::linearize([y, z]);

                let mut visible = self.visible_masks[face][i_2d];
                let upward_visible = self.visible_masks[face][i_2d + PADDED_STRIDE_Y_2D];
                let forward_visible = self.visible_masks[face][i_2d + PADDED_STRIDE_Z_2D];

                while visible != 0 {
                    let x = visible.trailing_zeros() as usize;
                    visible &= visible - 1;

                    let upward_i = Shape::linearize([x, 0, 0]);
                    let forward_i = Shape::linearize([x, y, 0]);

                    let pos = USizeVec3::new(x, y, z);
                    let voxel = view.get(pos.into());

                    // forward merging
                    if self.upward_merged[upward_i] == 0
                        && (forward_visible >> x) & 1 != 0
                        && context.can_merge(voxel, view.get(pos.with_z(pos.z + 1).into()))
                    {
                        self.forward_merged[forward_i] += 1;
                        continue;
                    }

                    // upward merging
                    if (upward_visible >> x) & 1 != 0
                        && self.forward_merged[forward_i]
                            == self.forward_merged[forward_i + STRIDE_Y]
                        && context.can_merge(voxel, view.get(pos.with_y(pos.y + 1).into()))
                    {
                        self.forward_merged[forward_i] = 0;
                        self.upward_merged[upward_i] += 1;
                        continue;
                    }

                    // finish
                    self.quads.push({
                        let forward_merged = self.forward_merged[forward_i] as u32;
                        let upward_merged = self.upward_merged[upward_i] as u32;

                        let x = x as u32;
                        let y = y as u32 - upward_merged;
                        let z = z as u32 - forward_merged;

                        let w = forward_merged + 1;
                        let h = upward_merged + 1;

                        let id = context.u26_shader_id(context.into_inner(voxel).unwrap(), face);

                        Quad::new([x, y, z], [w, h], [0; 4], id)
                    });

                    self.forward_merged[forward_i] = 0;
                    self.upward_merged[upward_i] = 0;
                }
            }
        }
    }

    fn merge_y<X, V, C>(&mut self, view: &V, context: &C, face: Face)
    where
        X: Copy,
        V: MesherView<Voxel = X>,
        C: MesherContext<Voxel = X>,
    {
        for z in 0..LEN {
            for y in 0..LEN {
                let i_2d = PaddedShape::linearize([y, z]);

                let mut visible = self.visible_masks[face][i_2d];
                let forward_visible = self.visible_masks[face][i_2d + PADDED_STRIDE_Z_2D];

                while visible != 0 {
                    let x = visible.trailing_zeros() as usize;

                    let forward_i = Shape::linearize([x, y, 0]);

                    let pos = USizeVec3::new(x, y, z);
                    let voxel = view.get(pos.into());

                    // forward merging
                    if (forward_visible >> x) & 1 != 0
                        && context.can_merge(voxel, view.get(pos.with_z(pos.z + 1).into()))
                    {
                        self.forward_merged[forward_i] += 1;
                        visible &= visible - 1;
                        continue;
                    }

                    // rightward merging
                    let mut i = 1;
                    while i < (LEN - x)
                        && (visible >> x >> i) & 1 != 0
                        && self.forward_merged[forward_i]
                            == self.forward_merged[forward_i + i * STRIDE_X]
                        && context.can_merge(voxel, view.get(pos.with_x(pos.x + 1).into()))
                    {
                        self.forward_merged[forward_i + i * STRIDE_X] = 0;
                        i += 1;
                    }
                    let right_merged = i;
                    visible &= !((1 << x << i) - 1);

                    // finish
                    self.quads.push({
                        let forward_merged = self.forward_merged[forward_i] as u32;

                        let x = x as u32;
                        let y = y as u32;
                        let z = z as u32 - forward_merged;

                        let w = right_merged as u32;
                        let h = forward_merged + 1;

                        let id = context.u26_shader_id(context.into_inner(voxel).unwrap(), face);

                        Quad::new([x, y, z], [w, h], [0; 4], id)
                    });

                    self.forward_merged[forward_i] = 0
                }
            }
        }
    }

    fn merge_z<X, V, C>(&mut self, view: &V, context: &C, face: Face)
    where
        X: Copy,
        V: MesherView<Voxel = X>,
        C: MesherContext<Voxel = X>,
    {
        for z in 0..LEN {
            for y in 0..LEN {
                let i_2d = PaddedShape::linearize([y, z]);

                let mut visible = self.visible_masks[face][i_2d];
                let upward_visible = self.visible_masks[face][i_2d + PADDED_STRIDE_Y_2D];

                while visible != 0 {
                    let x = visible.trailing_zeros() as usize;

                    let upward_i = Shape::linearize([x, 0, 0]);

                    let pos = USizeVec3::new(x, y, z);
                    let voxel = view.get(pos.into());

                    // upward merging
                    if (upward_visible >> x) & 1 != 0
                        && context.can_merge(voxel, view.get(pos.with_y(pos.y + 1).into()))
                    {
                        self.upward_merged[upward_i] += 1;
                        visible &= visible - 1;
                        continue;
                    }

                    // rightward merging
                    let mut i = 1;
                    while i < (LEN - x)
                        && (visible >> x >> i) & 1 != 0
                        && self.upward_merged[upward_i]
                            == self.upward_merged[upward_i + i * STRIDE_X]
                        && context.can_merge(voxel, view.get(pos.with_x(pos.x + 1).into()))
                    {
                        self.upward_merged[upward_i + i * STRIDE_X] = 0;
                        i += 1;
                    }
                    let right_merged = i;
                    visible &= !((1 << x << i) - 1);

                    // finish
                    self.quads.push({
                        let upward_merged = self.upward_merged[upward_i] as u32;

                        let x = x as u32;
                        let y = y as u32 - upward_merged;
                        let z = z as u32;

                        let w = right_merged as u32;
                        let h = upward_merged + 1;

                        let id = context.u26_shader_id(context.into_inner(voxel).unwrap(), face);

                        Quad::new([x, y, z], [w, h], [0; 4], id)
                    });

                    self.upward_merged[upward_i] = 0;
                }
            }
        }
    }
}
