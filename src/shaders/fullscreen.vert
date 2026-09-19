#version 450 core

out vec2 vUV;

// A fullscreen triangle conjured from gl_VertexID alone - no vertex buffer,
// no attributes, nothing bound. Drawn with glDrawArrays(GL_TRIANGLES, 0, 3).
//
//   id 0 -> uv (0,0) -> clip (-1,-1)
//   id 1 -> uv (2,0) -> clip ( 3,-1)
//   id 2 -> uv (0,2) -> clip (-1, 3)
//
// The triangle is deliberately larger than the screen; the bits that hang off
// the edges are clipped away. One triangle rather than two avoids a seam of
// duplicated fragment work along the diagonal.
void main()
{
    vUV = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);
}
