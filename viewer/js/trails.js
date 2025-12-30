/**
 * TrailManager Module
 * Handles high-performance rendering of particle trajectories
 */

import * as THREE from 'three';

export class TrailManager {
    constructor(scene, maxParticles = 1000, maxLength = 1000) {
        this.scene = scene;
        this.maxParticles = maxParticles;
        this.maxLength = maxLength;
        this.currentLength = 300; // Default active length for better visibility
        this.fadeAmount = 0.5;   // Default fade amount (0 to 1)
        this.enabled = false;

        this.initialized = false;
        this.mesh = null;
        this.history = []; // Array of arrays: [particleIdx][frameIdx]

        // Initialize history buffers
        for (let i = 0; i < maxParticles; i++) {
            this.history[i] = [];
        }

        this.initGeometry();
    }

    initGeometry() {
        // We use LineSegments. Each segment needs 2 vertices.
        // For a trail of length N, we have N-1 segments, so (N-1)*2 vertices.
        const vertexCount = this.maxParticles * (this.maxLength - 1) * 2;

        const geometry = new THREE.BufferGeometry();
        this.positions = new Float32Array(vertexCount * 3);
        this.opacities = new Float32Array(vertexCount);

        geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
        geometry.setAttribute('opacity', new THREE.BufferAttribute(this.opacities, 1));

        // Custom Shader to handle fade-out per vertex
        const material = new THREE.ShaderMaterial({
            transparent: true,
            uniforms: {
                color: { value: new THREE.Color(0x2196F3) },
                globalOpacity: { value: 1.0 }
            },
            vertexShader: `
                attribute float opacity;
                varying float vOpacity;
                void main() {
                    vOpacity = opacity;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 color;
                uniform float globalOpacity;
                varying float vOpacity;
                void main() {
                    gl_FragColor = vec4(color, vOpacity * globalOpacity);
                }
            `
        });

        this.mesh = new THREE.LineSegments(geometry, material);
        this.mesh.frustumCulled = false; // Prevent flickering when moving camera
        this.initialized = true;
    }

    setEnabled(enabled) {
        this.enabled = enabled;
        if (enabled) {
            this.scene.add(this.mesh);
        } else {
            this.scene.remove(this.mesh);
        }
    }

    setLength(n) {
        this.currentLength = Math.max(2, Math.min(this.maxLength, n));
        this.clearBuffers(); // Clear to prevent visual artifacts when stretching
    }

    setFade(fade) {
        this.fadeAmount = Math.max(0, Math.min(1, fade));
    }

    clear() {
        for (let i = 0; i < this.maxParticles; i++) {
            this.history[i] = [];
        }
        this.clearBuffers();
    }

    clearBuffers() {
        if (!this.initialized) return;
        this.positions.fill(0);
        this.opacities.fill(0);
        this.mesh.geometry.attributes.position.needsUpdate = true;
        this.mesh.geometry.attributes.opacity.needsUpdate = true;
    }

    update(particlePositions) {
        if (!this.enabled || !particlePositions || particlePositions.length === 0) return;

        const count = Math.min(particlePositions.length, this.maxParticles);
        const segmentsPerParticle = this.currentLength - 1;
        const maxSegments = this.maxLength - 1;

        // 1. Update history
        for (let i = 0; i < count; i++) {
            const pos = particlePositions[i];
            const pHistory = this.history[i];

            pHistory.unshift([pos[0], pos[1], pos[2]]);

            if (pHistory.length > this.currentLength) {
                pHistory.pop();
            }
        }

        // 2. Update buffer attributes with FIXED indexing
        for (let i = 0; i < this.maxParticles; i++) {
            const pHistory = this.history[i];
            const hLen = pHistory.length;
            const particleOffset = i * maxSegments * 2;

            for (let s = 0; s < maxSegments; s++) {
                const vIdx = particleOffset + s * 2;
                const v1_os = vIdx * 3;
                const v2_os = (vIdx + 1) * 3;

                // Only draw if within current length, history exists, AND segment is valid
                if (i < count && s < segmentsPerParticle && s < hLen - 1) {
                    const pos1 = pHistory[s];
                    const pos2 = pHistory[s + 1];

                    // Positions
                    this.positions[v1_os] = pos1[0];
                    this.positions[v1_os + 1] = pos1[1];
                    this.positions[v1_os + 2] = pos1[2];

                    this.positions[v2_os] = pos2[0];
                    this.positions[v2_os + 1] = pos2[1];
                    this.positions[v2_os + 2] = pos2[2];

                    // Opacities (linear power-fade)
                    const op1 = Math.pow(1.0 - (s / segmentsPerParticle), 2.0 * this.fadeAmount + 0.1);
                    const op2 = Math.pow(1.0 - ((s + 1) / segmentsPerParticle), 2.0 * this.fadeAmount + 0.1);

                    this.opacities[vIdx] = op1;
                    this.opacities[vIdx + 1] = op2;
                } else {
                    // CRITICAL: Explicitly clear unused segments in the buffer
                    this.positions[v1_os] = this.positions[v1_os + 1] = this.positions[v1_os + 2] = 0;
                    this.positions[v2_os] = this.positions[v2_os + 1] = this.positions[v2_os + 2] = 0;
                    this.opacities[vIdx] = 0;
                    this.opacities[vIdx + 1] = 0;
                }
            }
        }

        this.mesh.geometry.attributes.position.needsUpdate = true;
        this.mesh.geometry.attributes.opacity.needsUpdate = true;
    }
}
