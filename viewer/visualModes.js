// Visual Mode System for CosmoSim Viewer
// Provides color and size mappings based on entity properties

import * as THREE from 'three';

// Global visual mode state
export let VISUAL_MODE = 'mass-color';
export let HIDE_INACTIVE = false;

// Set visual mode
export function setVisualMode(mode) {
    VISUAL_MODE = mode;
}

// Set inactive hiding flag
export function setHideInactive(hide) {
    HIDE_INACTIVE = hide;
}

// ===================================================================
// COLOR MAPPING FUNCTIONS
// ===================================================================

/**
 * Map mass to color using smooth gradient
 * @param {number} mass - Entity mass
 * @param {number} minMass - Minimum mass in current frame
 * @param {number} maxMass - Maximum mass in current frame
 * @returns {THREE.Color}
 */
export function colorByMass(mass, minMass, maxMass) {
    // Normalize mass to 0-1 range
    const range = maxMass - minMass;
    const normalized = range > 0 ? (mass - minMass) / range : 0.5;

    // Define color stops
    const lowColor = new THREE.Color(50 / 255, 120 / 255, 255 / 255);  // Blue
    const midColor = new THREE.Color(80 / 255, 255 / 255, 80 / 255);   // Green
    const highColor = new THREE.Color(255 / 255, 80 / 255, 40 / 255);  // Red

    // Interpolate between colors
    if (normalized < 0.5) {
        // Low to mid
        return new THREE.Color().lerpColors(lowColor, midColor, normalized * 2);
    } else {
        // Mid to high
        return new THREE.Color().lerpColors(midColor, highColor, (normalized - 0.5) * 2);
    }
}

/**
 * Map entity type to discrete color
 * @param {number} type - Entity type index
 * @returns {THREE.Color}
 */
export function colorByType(type) {
    const palette = {
        0: new THREE.Color('steelblue'),
        1: new THREE.Color('gold'),
        2: new THREE.Color('crimson'),
        3: new THREE.Color('mediumseagreen'),
        4: new THREE.Color('purple'),
    };

    return palette[type] || new THREE.Color('gray');
}

/**
 * Map velocity magnitude to color
 * @param {Array} velocityVec - [vx, vy, vz]
 * @param {number} maxSpeed - Maximum speed in current frame
 * @returns {THREE.Color}
 */
export function colorByVelocity(velocityVec, maxSpeed) {
    // Calculate speed
    const [vx, vy, vz] = velocityVec;
    const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);

    // Normalize to 0-1
    const normalized = maxSpeed > 0 ? Math.min(speed / maxSpeed, 1) : 0;

    // Color gradient: dark blue -> cyan -> white
    const darkBlue = new THREE.Color(0x001030);
    const cyan = new THREE.Color(0x00FFFF);
    const white = new THREE.Color(0xFFFFFF);

    if (normalized < 0.5) {
        return new THREE.Color().lerpColors(darkBlue, cyan, normalized * 2);
    } else {
        return new THREE.Color().lerpColors(cyan, white, (normalized - 0.5) * 2);
    }
}

/**
 * Dim color if entity is inactive
 * @param {THREE.Color} baseColor - Base color
 * @param {boolean} activeFlag - Whether entity is active
 * @returns {THREE.Color}
 */
export function dimIfInactive(baseColor, activeFlag) {
    if (activeFlag) {
        return baseColor;
    }

    // Multiply RGB by 0.25 to dim
    return new THREE.Color(
        baseColor.r * 0.25,
        baseColor.g * 0.25,
        baseColor.b * 0.25
    );
}

/**
 * Calculate sphere radius based on mass
 * @param {number} mass - Entity mass
 * @returns {number} Radius
 */
export function radiusByMass(mass) {
    const baseRadius = 0.05;
    return baseRadius * (0.3 + Math.sqrt(mass));
}

/**
 * Get entity color based on current visual mode
 * @param {Object} entity - Entity data with mass, type, velocity, active
 * @param {Object} frameStats - Min/max values for normalization
 * @returns {THREE.Color}
 */
export function getEntityColor(entity, frameStats) {
    let color;

    switch (VISUAL_MODE) {
        case 'mass-color':
            color = colorByMass(entity.mass, frameStats.minMass, frameStats.maxMass);
            break;

        case 'type-color':
            color = colorByType(entity.type);
            break;

        case 'velocity-color':
            color = colorByVelocity(entity.velocity, frameStats.maxSpeed);
            break;

        case 'active-mask':
            color = entity.active ? new THREE.Color(0x55aaff) : new THREE.Color(0x222222);
            break;

        case 'uniform':
        default:
            color = new THREE.Color(0x55aaff);
            break;
    }

    // Apply inactive dimming if not in active-mask mode
    if (VISUAL_MODE !== 'active-mask') {
        color = dimIfInactive(color, entity.active);
    }

    return color;
}

/**
 * Calculate frame statistics for normalization
 * @param {Object} frame - Current frame data
 * @returns {Object} Stats with min/max values
 */
export function calculateFrameStats(frame) {
    const masses = frame.masses || [];
    const velocities = frame.velocities || [];
    const active = frame.active || [];

    // Calculate mass range
    let minMass = Infinity;
    let maxMass = -Infinity;
    for (let i = 0; i < masses.length; i++) {
        if (active[i]) {
            minMass = Math.min(minMass, masses[i]);
            maxMass = Math.max(maxMass, masses[i]);
        }
    }

    // Calculate max speed
    let maxSpeed = 0;
    for (let i = 0; i < velocities.length; i++) {
        if (active[i]) {
            const [vx, vy, vz] = velocities[i];
            const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
            maxSpeed = Math.max(maxSpeed, speed);
        }
    }

    return {
        minMass: isFinite(minMass) ? minMass : 1,
        maxMass: isFinite(maxMass) ? maxMass : 1,
        maxSpeed
    };
}
