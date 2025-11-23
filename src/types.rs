use std::f32::consts::FRAC_1_SQRT_2;
use std::fmt::Debug;

use Face::*;
use bit_iter::BitIter;
use enum_map::{Enum, EnumMap};

const MASK_6: u32 = (1 << 6) - 1;
const MASK_XYZ: u32 = (1 << 18) - 1;

// `Quad` & `Vertex`
const SHIFT_X: u32 = 0;
const SHIFT_Y: u32 = 6;
const SHIFT_Z: u32 = 12;

// `Quad`
const SHIFT_W: u32 = 18;
const SHIFT_H: u32 = 24;

// `Vertex`
const SHIFT_U: u32 = 18;
const SHIFT_V: u32 = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Enum)]
pub enum Face {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

#[derive(Debug)]
pub struct GreaterThanFive;

impl TryFrom<u8> for Face {
    type Error = GreaterThanFive;

    #[inline]
    fn try_from(value: u8) -> Result<Self, GreaterThanFive> {
        match value {
            0 => Ok(PosX),
            1 => Ok(NegX),
            2 => Ok(PosY),
            3 => Ok(NegY),
            4 => Ok(PosZ),
            5 => Ok(NegZ),
            _ => Err(GreaterThanFive),
        }
    }
}

impl Face {
    pub const ALL: [Self; 6] = [PosX, NegX, PosY, NegY, PosZ, NegZ];

    #[inline]
    pub const fn normal(self) -> [f32; 3] {
        match self {
            PosX => [1., 0., 0.],
            NegX => [-1., 0., 0.],
            PosY => [0., 1., 0.],
            NegY => [0., -1., 0.],
            PosZ => [0., 0., 1.],
            NegZ => [0., 0., -1.],
        }
    }
}

// TODO: Ambient Occlusion. Possibly steal 6 bits from `voxel_id` for 2 bit per vertex.
/// # Layout of `other`
/// x: 6 bits \
/// y: 6 bits \
/// z: 6 bits \
/// width (w): 6 bits \
/// height (h): 6 bits \
///
/// 0b00hh_hhhh_wwww_wwzz_zzzz_yyyy_yyxx_xxxx
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable, bytemuck::Pod))]
pub struct Quad {
    pub other: u32,
    pub id: u32,
}

impl Debug for Quad {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Quad")
            .field("position", &self.xyz())
            .field("size", &self.size())
            .field("id", &self.id)
            .finish()
    }
}

impl Quad {
    #[inline]
    pub const fn new(x: u32, y: u32, z: u32, w: u32, h: u32, id: u32) -> Self {
        Self {
            other: (h << SHIFT_H)
                | (w << SHIFT_W)
                | (z << SHIFT_Z)
                | (y << SHIFT_Y)
                | (x << SHIFT_X),
            id,
        }
    }

    #[inline]
    pub const fn x(self) -> u32 {
        (self.other >> SHIFT_X) & MASK_6
    }

    #[inline]
    pub const fn y(self) -> u32 {
        (self.other >> SHIFT_Y) & MASK_6
    }

    #[inline]
    pub const fn z(self) -> u32 {
        (self.other >> SHIFT_Z) & MASK_6
    }

    #[inline]
    pub const fn w(self) -> u32 {
        (self.other >> SHIFT_W) & MASK_6
    }

    #[inline]
    pub const fn h(self) -> u32 {
        (self.other >> SHIFT_H) & MASK_6
    }

    #[inline]
    pub const fn xyz(self) -> [u32; 3] {
        [self.x(), self.y(), self.z()]
    }

    #[inline]
    pub const fn size(self) -> [u32; 2] {
        [self.w(), self.h()]
    }

    #[inline]
    const fn packed_xyz(self) -> u32 {
        self.other & MASK_XYZ
    }

    pub const fn vertices(self, face: Face) -> [Vertex; 4] {
        let [w, h] = self.size();
        let xyz = self.packed_xyz();
        match face {
            Face::NegX => [
                Vertex::from_xyz_u_v(xyz, h, w),
                Vertex::from_xyz_u_v(xyz + packed_xyz(0, 0, h), 0, w),
                Vertex::from_xyz_u_v(xyz + packed_xyz(0, w, 0), h, 0),
                Vertex::from_xyz_u_v(xyz + packed_xyz(0, w, h), 0, 0),
            ],
            Face::NegY => [
                Vertex::from_xyz_u_v(xyz - packed_xyz(w, 0, 0) + packed_xyz(0, 0, h), w, h),
                Vertex::from_xyz_u_v(xyz - packed_xyz(w, 0, 0), w, 0),
                Vertex::from_xyz_u_v(xyz + packed_xyz(0, 0, h), 0, h),
                Vertex::from_xyz_u_v(xyz, 0, 0),
            ],
            Face::NegZ => [
                Vertex::from_xyz_u_v(xyz, w, h),
                Vertex::from_xyz_u_v(xyz + packed_xyz(0, h, 0), w, 0),
                Vertex::from_xyz_u_v(xyz + packed_xyz(w, 0, 0), 0, h),
                Vertex::from_xyz_u_v(xyz + packed_xyz(w, h, 0), 0, 0),
            ],
            Face::PosX => [
                Vertex::from_xyz_u_v(xyz, 0, 0),
                Vertex::from_xyz_u_v(xyz + packed_xyz(0, 0, h), h, 0),
                Vertex::from_xyz_u_v(xyz - packed_xyz(0, w, 0), 0, w),
                Vertex::from_xyz_u_v(xyz + packed_xyz(0, 0, h) - packed_xyz(0, w, 0), h, w),
            ],
            Face::PosY => [
                Vertex::from_xyz_u_v(xyz + packed_xyz(w, 0, h), w, h),
                Vertex::from_xyz_u_v(xyz + packed_xyz(w, 0, 0), w, 0),
                Vertex::from_xyz_u_v(xyz + packed_xyz(0, 0, h), 0, h),
                Vertex::from_xyz_u_v(xyz, 0, 0),
            ],
            Face::PosZ => [
                Vertex::from_xyz_u_v(xyz - packed_xyz(w, 0, 0) + packed_xyz(0, h, 0), 0, 0),
                Vertex::from_xyz_u_v(xyz - packed_xyz(w, 0, 0), 0, h),
                Vertex::from_xyz_u_v(xyz + packed_xyz(0, h, 0), w, 0),
                Vertex::from_xyz_u_v(xyz, w, h),
            ],
        }
    }

    pub fn transform(self, face: Face) -> ([f32; 3], [f32; 4], [f32; 3]) {
        fn as_f32(x: u32) -> f32 {
            x as f32
        }

        let [x, y, z] = self.xyz().map(as_f32);
        let [w, h] = self.size().map(as_f32);

        let scale = [w, h, 0.];

        let hw = w / 2.;
        let hh = h / 2.;

        let (translation, rotation) = match face {
            PosX => (
                [x + 1., y + hh, z + hw],
                [0., FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2],
            ),
            NegX => (
                [x + 0., y + hh, z + hw],
                [0., -FRAC_1_SQRT_2, 0., FRAC_1_SQRT_2],
            ),
            PosY => (
                [x + hw, y + 1., z + hh],
                [-FRAC_1_SQRT_2, 0., 0., FRAC_1_SQRT_2],
            ),
            NegY => (
                [x + hw, y + 0., z + hh],
                [FRAC_1_SQRT_2, 0., 0., FRAC_1_SQRT_2],
            ),
            PosZ => ([x + hw, y + hh, z + 1.], [0., 1., 0., 0.]),
            NegZ => ([x + hw, y + hh, z + 0.], [0., 0., 0., 1.]),
        };

        (scale, rotation, translation)
    }
}

#[inline]
const fn packed_xyz(x: u32, y: u32, z: u32) -> u32 {
    (z << SHIFT_Z) | (y << SHIFT_Y) | (x << SHIFT_X)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuadMesh(pub EnumMap<Face, Vec<Quad>>);

impl QuadMesh {
    #[inline]
    pub fn len(&self) -> usize {
        self.0.values().map(|vec| vec.len()).sum()
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshChanges {
    x: u64,
    y: u64,
    z: u64,
}

impl Debug for MeshChanges {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshChanges")
            .field("x", &self.xs().collect::<Vec<_>>())
            .field("y", &self.ys().collect::<Vec<_>>())
            .field("z", &self.zs().collect::<Vec<_>>())
            .finish()
    }
}

impl MeshChanges {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub const fn push(&mut self, [x, y, z]: [u32; 3]) {
        self.x |= 1 << x;
        self.y |= 1 << y;
        self.z |= 1 << z;
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.x == 0
    }

    #[inline]
    pub const fn clear(&mut self) {
        self.x = 0;
        self.y = 0;
        self.z = 0;
    }

    #[inline]
    pub fn xs(self) -> impl Iterator<Item = usize> {
        BitIter::from(self.x)
    }

    #[inline]
    pub fn ys(self) -> impl Iterator<Item = usize> {
        BitIter::from(self.y)
    }

    #[inline]
    pub fn zs(self) -> impl Iterator<Item = usize> {
        BitIter::from(self.z)
    }

    #[inline]
    pub fn to_array(self) -> [u64; 3] {
        self.into()
    }
}

impl From<MeshChanges> for [u64; 3] {
    #[inline]
    fn from(value: MeshChanges) -> Self {
        [value.x, value.y, value.z]
    }
}

/// # Layout
/// x: 6 bits \
/// y: 6 bits \
/// z: 6 bits \
/// u: 6 bits \
/// v: 6 bits \
///
/// 0b00vv_vvvv_uuuu_uuzz_zzzz_yyyy_yyxx_xxxx
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Vertex(pub u32);

impl Debug for Vertex {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Vertex")
            .field("position", &self.xyz())
            .field("u", &self.u())
            .field("v", &self.v())
            .finish()
    }
}

impl Vertex {
    #[inline]
    pub const fn new(x: u32, y: u32, z: u32, u: u32, v: u32) -> Self {
        Self((x << SHIFT_X) | (y << SHIFT_Y) | (z << SHIFT_Z) | (u << SHIFT_U) | (v << SHIFT_V))
    }

    #[inline]
    const fn from_xyz_u_v(xyz: u32, u: u32, v: u32) -> Self {
        Self((v << SHIFT_V) | (u << SHIFT_U) | (xyz << SHIFT_X))
    }

    #[inline]
    pub const fn x(self) -> u32 {
        (self.0 >> SHIFT_X) & MASK_6
    }

    #[inline]
    pub const fn y(self) -> u32 {
        (self.0 >> SHIFT_Y) & MASK_6
    }

    #[inline]
    pub const fn z(self) -> u32 {
        (self.0 >> SHIFT_Z) & MASK_6
    }

    #[inline]
    pub const fn u(self) -> u32 {
        (self.0 >> SHIFT_U) & MASK_6
    }

    #[inline]
    pub const fn v(self) -> u32 {
        (self.0 >> SHIFT_V) & MASK_6
    }

    #[inline]
    pub const fn xyz(self) -> [u32; 3] {
        [self.x(), self.y(), self.z()]
    }

    #[inline]
    pub const fn uv(self) -> [u32; 2] {
        [self.u(), self.v()]
    }
}
