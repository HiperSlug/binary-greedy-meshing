use std::fmt::Debug;

use bytemuck::{Pod, Zeroable};
use enum_map::Enum;
use glam::IVec3;

const MASK_26: u32 = (1 << 26) - 1;
const MASK_6: u32 = (1 << 6) - 1;
const MASK_2: u32 = (1 << 2) - 1;

const SHIFT_X: u32 = 0;
const SHIFT_Y: u32 = 6;
const SHIFT_Z: u32 = 12;
const SHIFT_W: u32 = 18;
const SHIFT_H: u32 = 24;
const SHIFT_AO_A: u32 = 30;
const SHIFT_ID: u32 = 0;
const SHIFT_AO_B: u32 = 26;
const SHIFT_AO_C: u32 = 28;
const SHIFT_AO_D: u32 = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[derive(Enum)]
pub enum Face {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl Face {
    pub const ALL: [Self; 6] = [
        Self::PosX,
        Self::NegX,
        Self::PosY,
        Self::NegY,
        Self::PosZ,
        Self::NegZ,
    ];

    pub const fn to_ivec3(self) -> IVec3 {
        match self {
            Self::PosX => IVec3::X,
            Self::NegX => IVec3::NEG_X,
            Self::PosY => IVec3::Y,
            Self::NegY => IVec3::NEG_Y,
            Self::PosZ => IVec3::Z,
            Self::NegZ => IVec3::NEG_Z,
        }
    }
}

/// # Contents
/// Holds a position offset inside it's chunk, a size, ambient occlusion, and the id of the voxel that created it
///
/// # Layout
/// x: 6 bits \
/// y: 6 bits \
/// z: 6 bits \
/// width (w): 6 bits \
/// height (h): 6 bits \
/// ao (o): 8 bits \
/// id (v): 26 bits \
///
/// [0baaaa_aavv_vvvv_vvvv_vvvv_vvvv_vvvv_vvvv, 0baahh_hhhh_wwww_wwzz_zzzz_yyyy_yyxx_xxxx]
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Zeroable, Pod)]
pub struct Quad([u32; 2]);

impl Debug for Quad {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Quad")
            .field("position", &self.xyz())
            .field("size", &self.size())
            .field("ao", &self.ao())
            .field("id", &self.id())
            .finish()
    }
}

impl Quad {
    pub const fn new(xyz: [u32; 3], size: [u32; 2], ao: [u32; 4], id: u32) -> Self {
        Self([
            ((xyz[0] & MASK_6) << SHIFT_X)
                | ((xyz[1] & MASK_6) << SHIFT_Y)
                | ((xyz[2] & MASK_6) << SHIFT_Z)
                | ((size[0] & MASK_6) << SHIFT_W)
                | ((size[1] & MASK_6) << SHIFT_H)
                | ((ao[0] & MASK_2) << SHIFT_AO_A),
            ((id & MASK_26) << SHIFT_ID)
                | ((ao[1] & MASK_2) << SHIFT_AO_B)
                | ((ao[2] & MASK_2) << SHIFT_AO_C)
                | ((ao[3] & MASK_2) << SHIFT_AO_D),
        ])
    }

    pub const fn x(self) -> u32 {
        (self.0[0] >> SHIFT_X) & MASK_6
    }

    pub const fn y(self) -> u32 {
        (self.0[0] >> SHIFT_Y) & MASK_6
    }

    pub const fn z(self) -> u32 {
        (self.0[0] >> SHIFT_Z) & MASK_6
    }

    pub const fn w(self) -> u32 {
        (self.0[0] >> SHIFT_W) & MASK_6
    }

    pub const fn h(self) -> u32 {
        (self.0[0] >> SHIFT_H) & MASK_6
    }

    pub const fn ao_a(self) -> u32 {
        (self.0[0] >> SHIFT_AO_A) & MASK_2
    }

    pub const fn ao_b(self) -> u32 {
        (self.0[1] >> SHIFT_AO_B) & MASK_2
    }

    pub const fn ao_c(self) -> u32 {
        (self.0[1] >> SHIFT_AO_C) & MASK_2
    }

    pub const fn ao_d(self) -> u32 {
        (self.0[1] >> SHIFT_AO_D) & MASK_2
    }

    pub const fn id(self) -> u32 {
        (self.0[1] >> SHIFT_ID) & MASK_26
    }

    pub const fn xyz(self) -> [u32; 3] {
        [self.x(), self.y(), self.z()]
    }

    pub const fn size(self) -> [u32; 2] {
        [self.w(), self.h()]
    }

    pub const fn ao(self) -> [u32; 4] {
        [self.ao_a(), self.ao_b(), self.ao_c(), self.ao_d()]
    }
}

/// This should be infallible and is restricted to within the meshed chunk
///
/// ## Speed
/// I suggest you `#[inline]` these functions
pub trait MesherView {
    type Voxel;

    fn get(&self, offset: [usize; 3]) -> Self::Voxel;
}

/// Tries to get the voxel in the adjacent (touching faces) chunk determined by `face` at the `offset`. If the chunk doesn't exist return None.
pub trait MesherViewAdjacent: MesherView {
    fn get_adjacent(&self, offset: [usize; 3], face: Face) -> Option<Self::Voxel>;
}

/// Tries to get the voxel in the neighboring (3x3x3 cube) chunk determined by `delta` at the `offset`. If the chunk doesn't exist return None.
pub trait MesherViewNeighborhood: MesherView {
    fn get_neighborhood(&self, offset: [usize; 3], delta: [i32; 3]) -> Option<Self::Voxel>;
}

/// Determines which voxels exist, are visible, are merged, and how they are represented in shaders.
pub trait MesherContext {
    type Voxel;
    type InnerVoxel;

    fn into_inner(&self, voxel: Self::Voxel) -> Option<Self::InnerVoxel>;

    fn is_visible(&self, voxel: Self::InnerVoxel, adj_voxel: Self::InnerVoxel) -> bool;

    fn can_merge(&self, voxel: Self::Voxel, adj_voxel: Self::Voxel) -> bool;

    fn u26_shader_id(&self, voxel: Self::InnerVoxel, face: Face) -> u32;
}
