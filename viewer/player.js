// Frame Player Module for CosmoSim Viewer
// Handles playback control and frame updates

import * as THREE from 'three';
import * as VisualModes from './visualModes.js';
import { TopologyOverlay } from './topologyOverlay.js';

export class FramePlayer {
    constructor(frames, scene) {
        this.frames = frames;
        this.scene = scene;
        this.currentFrame = 0;
        this.isPlaying = false;
        this.fps = 30;
        this.lastFrameTime = 0;
        this.instancedMesh = null;
        this.dummy = new THREE.Object3D(); // Helper for matrix calculations

        // Topology Overlay
        this.topologyOverlay = new TopologyOverlay(scene);

        console.log(`FramePlayer initialized with ${frames.length} frames`);
    }

    /**
     * Create Three.js InstancedMesh for all entities
     */
    initializeEntities() {
        if (this.frames.length === 0) return;

        // Find max entity count across all frames to allocate buffer
        // (Assuming mostly constant, but safe to check first frame or max)
        const firstFrame = this.frames[0];
        const count = firstFrame.positions ? firstFrame.positions.length : 0;

        // Clear existing mesh
        if (this.instancedMesh) {
            this.scene.remove(this.instancedMesh);
            this.instancedMesh.geometry.dispose();
            this.instancedMesh.material.dispose();
        }

        // Create geometry and material
        // Base radius 1.0, we'll scale it via matrix
        const geometry = new THREE.SphereGeometry(1.0, 16, 16);
        const material = new THREE.MeshPhongMaterial({
            color: 0xffffff, // White base, tinted by instanceColor
            flatShading: false
        });

        this.instancedMesh = new THREE.InstancedMesh(geometry, material, count);
        this.instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        this.instancedMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);

        this.scene.add(this.instancedMesh);
        console.log(`Created InstancedMesh with ${count} instances`);

        // Initial update
        this.updateFrame();
    }

    /**
     * Update entity positions and visuals from current frame
     */
    updateFrame() {
        if (this.frames.length === 0 || !this.instancedMesh) {
            return;
        }

        // Wrap frame index if needed (though update() handles this)
        if (this.currentFrame >= this.frames.length) this.currentFrame = 0;

        const frame = this.frames[this.currentFrame];
        const positions = frame.positions || [];
        const masses = frame.masses || [];
        const types = frame.types || [];
        const velocities = frame.velocities || [];
        const active = frame.active || [];

        // Update Topology Overlay
        if (frame.topology) {
            this.topologyOverlay.update(frame.topology);
        }

        // Calculate frame stats for normalization
        const stats = VisualModes.calculateFrameStats(frame);

        let activeCount = 0;

        for (let i = 0; i < this.instancedMesh.count; i++) {
            if (i >= positions.length) {
                // Hide extra instances
                this.dummy.scale.set(0, 0, 0);
                this.dummy.updateMatrix();
                this.instancedMesh.setMatrixAt(i, this.dummy.matrix);
                continue;
            }

            const isActive = active[i];

            // Handle visibility
            if (!isActive && VisualModes.VISUAL_MODE === 'active-mask') {
                // Hide completely in active-mask mode
                this.dummy.scale.set(0, 0, 0);
                this.dummy.updateMatrix();
                this.instancedMesh.setMatrixAt(i, this.dummy.matrix);
                continue;
            }

            // Position
            const pos = positions[i];
            this.dummy.position.set(pos[0], pos[1], pos[2]);

            // Scale (Radius)
            const mass = masses[i] || 1.0;
            const radius = VisualModes.radiusByMass(mass);
            this.dummy.scale.set(radius, radius, radius);

            this.dummy.updateMatrix();
            this.instancedMesh.setMatrixAt(i, this.dummy.matrix);

            // Color
            const entity = {
                mass: mass,
                type: types[i] || 0,
                velocity: velocities[i] || [0, 0, 0],
                active: isActive
            };

            const color = VisualModes.getEntityColor(entity, stats);
            this.instancedMesh.setColorAt(i, color);

            if (isActive) activeCount++;
        }

        this.instancedMesh.instanceMatrix.needsUpdate = true;
        if (this.instancedMesh.instanceColor) this.instancedMesh.instanceColor.needsUpdate = true;
    }

    /**
     * Start playback
     */
    play() {
        this.isPlaying = true;
        this.lastFrameTime = performance.now();
    }

    /**
     * Pause playback
     */
    pause() {
        this.isPlaying = false;
    }

    /**
     * Toggle play/pause
     */
    toggle() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    /**
     * Restart from frame 0
     */
    restart() {
        this.currentFrame = 0;
        this.updateFrame();
        this.pause();
    }

    /**
     * Step to next frame
     */
    next() {
        if (this.currentFrame < this.frames.length - 1) {
            this.currentFrame++;
            this.updateFrame();
        }
    }

    /**
     * Step to previous frame
     */
    prev() {
        if (this.currentFrame > 0) {
            this.currentFrame--;
            this.updateFrame();
        }
    }

    /**
     * Update player state (call every animation frame)
     * @param {number} currentTime - Current timestamp from requestAnimationFrame
     */
    update(currentTime) {
        if (!this.isPlaying || this.frames.length === 0) {
            return;
        }

        const frameDuration = 1000 / this.fps;
        const elapsed = currentTime - this.lastFrameTime;

        if (elapsed >= frameDuration) {
            this.lastFrameTime = currentTime;

            // Advance to next frame
            this.currentFrame++;

            // Loop back to start if at end
            if (this.currentFrame >= this.frames.length) {
                this.currentFrame = 0;
            }

            this.updateFrame();
        }
    }

    /**
     * Get current frame info for HUD
     */
    getFrameInfo() {
        if (this.frames.length === 0) {
            return { frame: 0, total: 0, time: 0, activeCount: 0 };
        }

        const frame = this.frames[this.currentFrame];
        const activeCount = (frame.active || []).filter(a => a).length;
        const time = frame.time || 0;

        return {
            frame: this.currentFrame,
            total: this.frames.length,
            time: time,
            activeCount: activeCount
        };
    }

    /**
     * Set playback speed (frames per second)
     */
    setFPS(fps) {
        this.fps = Math.max(1, Math.min(60, fps));
    }

    /**
     * Force visual update (e.g. when mode changes)
     */
    refreshVisuals() {
        this.updateFrame();
    }
}
