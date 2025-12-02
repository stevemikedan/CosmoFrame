// Auto Camera Module for CosmoSim Viewer
// Automatically adjusts camera position and zoom to keep simulation in view

import * as THREE from 'three';

/**
 * Linear interpolation between two values
 * @param {number} a - Start value
 * @param {number} b - End value
 * @param {number} t - Interpolation factor (0-1)
 * @returns {number} - Interpolated value
 */
export function lerp(a, b, t) {
    return a + (b - a) * t;
}

/**
 * Linear interpolation between two THREE.Vector3 objects
 * @param {THREE.Vector3} v1 - Start vector
 * @param {THREE.Vector3} v2 - End vector
 * @param {number} t - Interpolation factor (0-1)
 * @returns {THREE.Vector3} - Interpolated vector
 */
function lerpVector3(v1, v2, t) {
    return new THREE.Vector3(
        lerp(v1.x, v2.x, t),
        lerp(v1.y, v2.y, t),
        lerp(v1.z, v2.z, t)
    );
}

/**
 * Compute bounding box from entity positions
 * @param {Array<Array<number>>} positions - Array of [x, y, z] position arrays
 * @returns {{min: Array<number>, max: Array<number>}} - Bounding box
 */
export function computeBoundingBox(positions) {
    // Robustness check
    if (!Array.isArray(positions) || positions.length === 0) {
        return {
            min: [0, 0, 0],
            max: [0, 0, 0]
        };
    }

    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];

    for (const pos of positions) {
        if (!pos || pos.length < 3) continue;

        for (let i = 0; i < 3; i++) {
            if (pos[i] < min[i]) min[i] = pos[i];
            if (pos[i] > max[i]) max[i] = pos[i];
        }
    }

    // Handle case where all positions are invalid
    if (!isFinite(min[0])) {
        return {
            min: [0, 0, 0],
            max: [0, 0, 0]
        };
    }

    return { min, max };
}

/**
 * Compute bounding sphere from bounding box
 * @param {{min: Array<number>, max: Array<number>}} bbox - Bounding box
 * @returns {{center: Array<number>, radius: number}} - Bounding sphere
 */
export function computeBoundingSphere(bbox) {
    const { min, max } = bbox;

    // Compute center
    const center = [
        (min[0] + max[0]) / 2,
        (min[1] + max[1]) / 2,
        (min[2] + max[2]) / 2
    ];

    // Compute radius (half the diagonal of the bounding box)
    const dx = max[0] - min[0];
    const dy = max[1] - min[1];
    const dz = max[2] - min[2];
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;

    return { center, radius };
}

// State for smooth radius tracking
let smoothedRadius = null;

/**
 * Update camera view to keep all entities in frame
 * @param {THREE.Camera} camera - Three.js camera
 * @param {Array<Array<number>>} positions - Array of entity positions
 * @param {Object} config - Configuration object with cameraPadding
 */
export function updateCameraView(camera, positions, config) {
    // Robustness check
    if (!Array.isArray(positions) || positions.length === 0) {
        return;
    }

    // Compute bounding sphere
    const bbox = computeBoundingBox(positions);
    const sphere = computeBoundingSphere(bbox);
    const { center, radius } = sphere;

    // Safety check for invalid radius
    if (!isFinite(radius) || radius <= 0) {
        return;
    }

    // Initialize smoothed radius on first call
    if (smoothedRadius === null) {
        smoothedRadius = radius;
    }

    // Smooth radius to avoid jitter from oscillating particles
    smoothedRadius = lerp(smoothedRadius, radius, 0.1);

    // Cap maximum radius for extreme cases
    const MAX_RADIUS = 1e6;
    const cappedRadius = Math.min(smoothedRadius, MAX_RADIUS);

    // Calculate desired camera distance with padding
    const padding = config.cameraPadding || 1.4;
    const desiredDistance = cappedRadius * padding;

    // Current camera direction (normalized)
    const currentPos = camera.position.clone();
    const currentTarget = new THREE.Vector3(...center);
    const direction = currentPos.clone().sub(currentTarget).normalize();

    // If direction is zero (camera at center), use default direction
    if (direction.length() < 0.001) {
        direction.set(1, 1, 1).normalize();
    }

    // Desired camera position
    const desiredPos = new THREE.Vector3(...center).add(
        direction.multiplyScalar(desiredDistance)
    );

    // Smoothly interpolate camera position
    const newPos = lerpVector3(currentPos, desiredPos, 0.05);
    camera.position.copy(newPos);

    // Smoothly interpolate look-at target
    const currentLookAt = new THREE.Vector3(0, 0, 0); // Approximate current target
    const desiredLookAt = new THREE.Vector3(...center);
    const newLookAt = lerpVector3(currentLookAt, desiredLookAt, 0.05);
    camera.lookAt(newLookAt);
}

/**
 * Reset camera to default position and orientation
 * @param {THREE.Camera} camera - Three.js camera
 * @param {Object} controls - OrbitControls instance
 * @param {Object} config - Configuration object with defaultCamera settings
 */
export function resetCamera(camera, controls, config) {
    const defaults = config.defaultCamera || {
        position: [3, 3, 6],
        target: [0, 0, 0]
    };

    // Reset camera position
    camera.position.set(...defaults.position);

    // Reset controls target
    if (controls) {
        controls.target.set(...defaults.target);
        controls.update();
    }

    // Reset look-at
    camera.lookAt(...defaults.target);

    // Reset smoothed radius
    smoothedRadius = null;

    console.log('Camera reset to default position');
}
