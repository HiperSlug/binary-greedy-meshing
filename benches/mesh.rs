use std::collections::HashMap;
use std::hint::black_box;

use binary_greedy_meshing as bgm;
use criterion::{Criterion, criterion_group, criterion_main};
use enum_map::EnumMap;
use glam::IVec3;
use ndshape::ConstShape;

struct Blocks(Vec<bool>);

impl bgm::MesherContext for Blocks {
    type Voxel = u16;
    type InnerVoxel = u16;

    #[inline]
    fn into_inner(&self, voxel: Self::Voxel) -> Option<Self::InnerVoxel> {
        (voxel != u16::MAX).then_some(voxel)
    }

    #[inline]
    fn is_visible(&self, voxel: Self::Voxel, adj_voxel: Self::Voxel) -> bool {
        voxel != adj_voxel && self.0[voxel as usize]
    }

    #[inline]
    fn can_merge(&self, voxel: Self::Voxel, adj_voxel: Self::Voxel) -> bool {
        voxel == adj_voxel
    }

    #[inline]
    fn u26_shader_id(&self, voxel: Self::InnerVoxel, _face: bgm::Face) -> u32 {
        voxel as u32
    }
}

struct Chunk([u16; bgm::CUBE]);

struct View<'a> {
    main: &'a Chunk,
    adj: EnumMap<bgm::Face, Option<&'a Chunk>>,
}

impl<'a> View<'a> {
    pub fn new(map: &'a HashMap<IVec3, Chunk>, pos: IVec3) -> Option<Self> {
        Some(Self {
            main: map.get(&pos)?,
            adj: EnumMap::from_fn(|face: bgm::Face| map.get(&(pos + face.to_ivec3()))),
        })
    }
}

impl<'a> bgm::MesherView for View<'a> {
    type Voxel = u16;

    #[inline]
    fn get(&self, offset: [usize; 3]) -> Self::Voxel {
        let index = bgm::Shape::linearize(offset);
        self.main.0[index]
    }
}

impl<'a> bgm::MesherViewAdjacent for View<'a> {
    #[inline]
    fn get_adjacent(&self, offset: [usize; 3], face: bgm::Face) -> Option<Self::Voxel> {
        let index = bgm::Shape::linearize(offset);
        self.adj[face].map(|c| c.0[index])
    }
}

fn init() -> (HashMap<IVec3, Chunk>, Blocks) {
    let context = Blocks(vec![false]);

    let mut map = HashMap::new();
    let chunk = map
        .entry(IVec3::ZERO)
        .or_insert(Chunk([u16::MAX; bgm::CUBE]));

    for x in 0..bgm::LEN {
        for y in 0..bgm::LEN {
            for z in 0..bgm::LEN {
                let i_3d = bgm::Shape::linearize([x, y, z]);
                if inside_sphere([x as u32 - 31, y as u32 - 31, z as u32 - 31], 16) {
                    chunk.0[i_3d] = 0;
                }
            }
        }
    }

    (map, context)
}

fn inside_sphere(pos: [u32; 3], radius: u32) -> bool {
    let length_squared = pos.into_iter().fold(0, |fold, elem| fold + elem * elem);
    length_squared < radius.pow(2)
}

fn mesh(c: &mut Criterion) {
    let (map, context) = black_box(init());
    let view = View::new(&map, IVec3::ZERO).unwrap();

    let mut mesher = bgm::Mesher::new();

    c.bench_function("mesh", |b| {
        b.iter(|| {
            mesher.mesh(&view, &context);
        });
    });
}

criterion_group!(mesh_group, mesh);
criterion_main!(mesh_group);
