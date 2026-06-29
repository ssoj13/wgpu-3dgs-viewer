import package::camera::{ Camera, world_to_camera };
import wgpu_3dgs_core::{
    gaussian::{
        Gaussian,
        gaussian_unpack_color,
        gaussian_unpack_cov3d,
        gaussian_unpack_sh,
    },
    model_transform::{
        ModelTransform,
        model_to_world,
        model_transform_mat,
        model_scale_rot_mat,
    },
};

// Determine if a Gaussian should be culled.
fn cull(ndc_pos: vec3<f32>) -> bool {
    return !(all(ndc_pos >= vec3<f32>(-1.0, -1.0, 0.0)) && all(ndc_pos <= vec3<f32>(1.0)));
}

// Project a 3D covariance into a 2D (screen-space) covariance (pixel²), WITHOUT the low-pass
// dilation. Faithful port of bevy_gaussian_splatting `helpers::cov2d` / inria 3DGS, adapted to this
// crate's Camera (view = view_from_world, proj = clip_from_view, size = viewport px).
//
// CRITICAL: `focal` uses the FULL viewport size (proj-NDC-scale × size, no ×0.5). A standard
// projection has proj[0][0] = 2·f/width, so focal = 2·f_px — exactly bevy's units. Combined with the
// ½-NDC mapping in render.wesl this yields the correct 3σ footprint in REAL pixels, which is what
// makes the +0.3 dilation cover ≳ 1px and anti-alias cleanly. The previous ×0.5 focal left the
// covariance in half-pixel² units, so +0.3 under-covered (≈ 0.27 real px) → sub-pixel sparkle.
fn cov2d_raw(gaussian: Gaussian, model_transform: ModelTransform, camera: Camera) -> vec3<f32> {
    let cov3d = gaussian_unpack_cov3d(gaussian);
    let sr = model_scale_rot_mat(model_transform);

    let vrk = mat3x3<f32>(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5],
    );

    // This crate's ORIGINAL EWA Jacobian / orientation (proven correct for this camera convention,
    // i.e. +focal.y). The ONLY change from the original is the focal scale: full viewport size
    // instead of size*0.5, so the covariance is in real pixels (proj[0][0] = 2f/width -> focal = 2f,
    // bevy's units). That, with the half-NDC placement in render.wesl, gives the correct 3-sigma
    // footprint AND makes the +0.3 low-pass dilation cover >=1px (clean AA). The old *0.5 left it in
    // half-pixel units, so +0.3 under-covered (~0.27 real px) -> sub-pixel sparkle/grain.
    let focal = vec2<f32>(camera.proj[0][0], camera.proj[1][1]) * camera.size;

    let t = camera.view * model_transform_mat(model_transform) * vec4<f32>(gaussian.pos, 1.0);
    let j = transpose(mat3x3<f32>(
        focal.x / t.z, 0.0, -(focal.x * t.x) / (t.z * t.z),
        0.0, focal.y / t.z, -(focal.y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0,
    ));
    let w = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);

    let m = j * w * sr;
    let cov = m * vrk * transpose(m);

    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}

// Screen-space low-pass dilation (px²) added to the 2D-covariance diagonal (inria 3DGS / bevy):
// guarantees every splat covers ≳ 1px so sub-pixel gaussians stop aliasing/sparkling.
const cov2d_dilation = 0.3;

// Projected 2D covariance WITH the low-pass dilation applied.
fn cov2d(gaussian: Gaussian, model_transform: ModelTransform, camera: Camera) -> vec3<f32> {
    let c = cov2d_raw(gaussian, model_transform, camera);
    return vec3<f32>(c.x + cov2d_dilation, c.y, c.z + cov2d_dilation);
}

// Inverse of the (dilated) 2D covariance — the "conic" used to evaluate the gaussian per pixel:
// power = -0.5·(conic.x·dx² + conic.z·dy²) + conic.y·dx·dy   (bevy GAUSSIAN_3D fragment path).
fn cov2d_conic(cov: vec3<f32>) -> vec3<f32> {
    let det = cov.x * cov.z - cov.y * cov.y;
    let det_inv = 1.0 / det;
    return vec3<f32>(cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv);
}

// Calculate the diagonal axes of a 2D covariance matrix.
//
// The first two components is the major axis, last two components is the minor axis.
// If the returned vector is (0, 0, 0, 0), it means the Gaussian is not visible.
fn cov2d_axes(
    gaussian: Gaussian,
    model_transform: ModelTransform,
    camera: Camera,
    std_dev: f32,
) -> vec4<f32> {
    let cov2d = cov2d(gaussian, model_transform, camera);
    let mid = 0.5 * (cov2d.x + cov2d.z);
    let radius = length(vec2<f32>(0.5 * (cov2d.x - cov2d.z), cov2d.y));
    let major_lambda = mid + radius;
    let minor_lambda = mid - radius;

    if minor_lambda < 0.0 {
        return vec4<f32>(0.0);
    }

    let diag = vec2<f32>(cov2d.y, major_lambda - cov2d.x);
    let diag_dir = select(normalize(diag), vec2<f32>(0.0, 1.0), all(diag == vec2<f32>(0.0)));
    let ortho_diag_dir = vec2<f32>(diag_dir.y, -diag_dir.x);
    let major_len = min(std_dev * sqrt(major_lambda), 1024.0);
    let minor_len = min(std_dev * sqrt(minor_lambda), 1024.0);
    let major_axis = major_len * diag_dir;
    let minor_axis = minor_len * ortho_diag_dir;

    return vec4<f32>(major_axis, minor_axis);
}

// Calculate the color of a Gaussian based on its SH coefficients and view direction.
fn view_color(gaussian: Gaussian, dir: vec3<f32>, sh_deg: u32, no_sh0: bool) -> vec4<f32> {
    const sh_c1 = 0.4886025;
    const sh_c2 = array<f32, 5>(1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742);
    const sh_c3 = array<f32, 7>(
        -0.5900436, 2.8906114, -0.4570458, 0.3731763, -0.4570458, 1.4453057, -0.5900436
    );

    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    let c = gaussian_unpack_color(gaussian);
    var result = c.rgb; // 0.5 + SH_C0 * sh[0] already precomputed

    if no_sh0 {
        result = vec3<f32>(0.5);
    }

    if sh_deg >= 1u {
        result += sh_c1 * (
            -gaussian_unpack_sh(gaussian, 0u) * y +
            gaussian_unpack_sh(gaussian, 1u) * z -
            gaussian_unpack_sh(gaussian, 2u) * x
        );

        if sh_deg >= 2u {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;

            result += 
                sh_c2[0] * xy * gaussian_unpack_sh(gaussian, 3u) +
                sh_c2[1] * yz * gaussian_unpack_sh(gaussian, 4u) +
                sh_c2[2] * (2.0 * zz - xx - yy) * gaussian_unpack_sh(gaussian, 5u) +
                sh_c2[3] * xz * gaussian_unpack_sh(gaussian, 6u) +
                sh_c2[4] * (xx - yy) * gaussian_unpack_sh(gaussian, 7u);

            if sh_deg >= 3u {
                result += 
                    sh_c3[0] * y * (3.0 * xx - yy) * gaussian_unpack_sh(gaussian, 8u) +
                    sh_c3[1] * xy * z * gaussian_unpack_sh(gaussian, 9u) +
                    sh_c3[2] * y * (4.0 * zz - xx - yy) * gaussian_unpack_sh(gaussian, 10u) +
                    sh_c3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * gaussian_unpack_sh(gaussian, 11u) +
                    sh_c3[4] * x * (4.0 * zz - xx - yy) * gaussian_unpack_sh(gaussian, 12u) +
                    sh_c3[5] * z * (xx - yy) * gaussian_unpack_sh(gaussian, 13u) +
                    sh_c3[6] * x * (xx - 3.0 * yy) * gaussian_unpack_sh(gaussian, 14u);
            }
        }
    }

    return vec4<f32>(max(result, vec3<f32>(0.0)), c.a);
}