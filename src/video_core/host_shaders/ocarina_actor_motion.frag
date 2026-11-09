#version 450
#define OCARINA_COMPARE_SCENE_DEPTH 1

layout(location = 0) in vec4 previous_clip;
layout(location = 0) out vec2 motion;

#ifdef OCARINA_COMPARE_SCENE_DEPTH
layout(binding = 3) uniform sampler2D scene_depth;
layout(push_constant) uniform DepthComparison {
    uint depth_tolerance;
    // Allowed difference in distance, as a fraction of it. Exact bit matches only held on the GPU
    // the replay was captured on: AMD's compiler rounds the game's depth differently.
    float distance_tolerance;
} comparison;
#endif

layout(std140, binding = 2, row_major) uniform Transforms {
    mat4 current_view_projection;
    mat4 previous_view_projection;
    vec4 current_origin;
    vec4 previous_origin;
    vec2 extent;
} transforms;

void main() {
#ifdef OCARINA_COMPARE_SCENE_DEPTH
    uint expected = floatBitsToUint(texelFetch(scene_depth, ivec2(gl_FragCoord.xy), 0).r);
    uint actual = floatBitsToUint(gl_FragCoord.z);
    if (max(expected, actual) - min(expected, actual) > comparison.depth_tolerance) {
        // Depth grows with distance as 1 - near / distance, so 1 - depth is near / distance and
        // their ratio compares distances.
        float expected_inverse = 1.0 - uintBitsToFloat(expected);
        float actual_inverse = 1.0 - gl_FragCoord.z;
        // The vertex shader maps [-1, 1] clip depth to [0, 1]; on the AMD test machine the scene's
        // depth matched only Link's unmapped depth (1 - depth half the scene's), so either counts.
        float unmapped_inverse = 2.0 * actual_inverse;
        float allowed = comparison.distance_tolerance * max(expected_inverse, 1e-12);
        if (abs(expected_inverse - actual_inverse) > allowed &&
            abs(expected_inverse - unmapped_inverse) > allowed) {
            discard;
        }
    }
#endif
    if (previous_clip.w <= 1e-6) {
        discard;
    }
    vec2 previous_uv = previous_clip.xy / previous_clip.w * vec2(0.5, -0.5) + 0.5;
    vec2 displacement = previous_uv * transforms.extent - gl_FragCoord.xy;
    if (!all(lessThanEqual(abs(displacement), vec2(65504)))) {
        discard;
    }
    motion = displacement;
}
