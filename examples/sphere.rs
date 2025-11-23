use std::collections::HashSet;

use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::prelude::*;
use binary_greedy_meshing as bgm;

fn main() {
    App::new()
        .init_resource::<WireframeConfig>()
        .add_plugins((
            DefaultPlugins,
            WireframePlugin::default(),
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut wireframe_config: ResMut<WireframeConfig>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    wireframe_config.global = true;

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: light_consts::lux::OVERCAST_DAY,
        ..Default::default()
    });

    commands.spawn((
        Transform::from_translation(vec3(50.0, 100.0, 50.0)),
        PointLight {
            range: 200.0,
            //intensity: 8000.0,
            ..Default::default()
        },
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(vec3(60.0, 60.0, 100.0))
            .looking_at(vec3(31.0, 31.0, 31.0), Vec3::Y),
    ));

    let rectangle = meshes.add(Rectangle::from_length(1.));
    let color = materials.add(Color::linear_rgba(0.1, 0.1, 0.1, 1.0));

    for (face, quads) in &generate_mesh().0 {
        for quad in quads {
            let (scale, rotation, translation) = quad.transform(face);
            commands.spawn((
                Mesh3d(rectangle.clone()),
                MeshMaterial3d(color.clone()),
                Transform {
                    scale: scale.into(),
                    rotation: Quat::from_array(rotation),
                    translation: translation.into(),
                },
            ));
        }
    }
}

fn generate_mesh() -> bgm::QuadMesh {
    let transparents = HashSet::new();

    let voxels = voxel_buffer();
    let opaque_masks = bgm::compute_opaque_masks(&voxels, &transparents);
    let transparent_masks = bgm::compute_transparent_masks(&voxels, &transparents);

    let mut mesher = bgm::Mesher::new();

    mesher.mesh(&voxels, &opaque_masks, &transparent_masks)
}

fn voxel_buffer() -> [u16; bgm::CUBE] {
    let mut voxels = [0; bgm::CUBE];
    for x in 0..bgm::LEN {
        for y in 0..bgm::LEN {
            for z in 0..bgm::LEN {
                let i_3d = bgm::linearize_3d(x, y, z);
                let pos = uvec3(x as u32, y as u32, z as u32);
                let origin = uvec3(31, 31, 31);
                let radius = 16;
                voxels[i_3d] = inside_sphere(pos, origin, radius) as u16;
            }
        }
    }
    voxels
}

fn inside_sphere(pos: UVec3, origin: UVec3, radius: u32) -> bool {
    (pos - origin).length_squared() > radius.pow(2)
}
