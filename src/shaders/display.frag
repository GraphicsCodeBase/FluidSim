#version 450 core

in  vec2 vUV;
out vec4 fragColor;

uniform sampler2D uField;

// Phase 0: show the texture as-is.
//
// Note the vertical convention. Texture row 0 is sampled at v = 0, which is
// the BOTTOM of the screen. That is deliberate: it makes grid +y point up, so
// buoyancy in Phase 3 is simply +y and needs no sign flips. If an image ever
// looks upside down, this is why.
void main()
{
    fragColor = texture(uField, vUV);
}
