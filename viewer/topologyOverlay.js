// Topology Overlay Module for CosmoSim Viewer
// Renders visualization of the universe topology

import * as THREE from 'three';

export class TopologyOverlay {
    constructor(scene) {
        this.scene = scene;
        this.group = new THREE.Group();
        this.scene.add(this.group);

        this.currentKey = null;
        this.isVisible = true;
        this.currentMode = 'none';

        // Callback for UI updates
        this.onModeChange = null;
    }

    /**
     * Update topology visualization based on frame data
     * @param {Object} topologyData - Topology object from frame JSON
     */
    update(topologyData) {
        const mode = topologyData?.mode || 'unknown';
        const params = topologyData?.params || {};
        const bounds = typeof params.bounds === 'number' ? params.bounds : 1.0;
        const radius = typeof params.radius === 'number' ? params.radius : bounds;

        // Generate key for change detection
        const newKey = `${mode}:${bounds}:${radius}`;

        if (newKey !== this.currentKey) {
            this.rebuild(mode, bounds, radius);
            this.currentKey = newKey;
            this.currentMode = mode;

            if (this.onModeChange) {
                this.onModeChange(mode);
            }
        }
    }

    /**
     * Rebuild topology geometry
     */
    rebuild(mode, bounds, radius) {
        // Clear existing children
        while (this.group.children.length > 0) {
            const child = this.group.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(m => m.dispose());
                } else {
                    child.material.dispose();
                }
            }
            this.group.remove(child);
        }

        if (mode === 'none') return;

        switch (mode) {
            case 'flat':
                this.buildFlat(bounds);
                break;
            case 'bubble':
                this.buildBubble(radius);
                break;
            case 'toroidal':
                this.buildToroidal(bounds);
                break;
            case 'lattice2d':
                this.buildLattice2D(bounds);
                break;
            case 'lattice3d':
                this.buildLattice3D(bounds);
                break;
            case 'unknown':
            default:
                this.buildUnknown(bounds);
                break;
        }
    }

    /**
     * Set visibility of the overlay
     */
    setVisible(visible) {
        this.isVisible = visible;
        this.group.visible = visible;
    }

    // --- Geometry Builders ---

    buildFlat(bounds) {
        const size = bounds * 2;
        const divisions = 20;
        const gridHelper = new THREE.GridHelper(size, divisions, 0x444444, 0x222222);
        // GridHelper is on XZ plane by default. If simulation is XY, we might need to rotate.
        // CosmoSim usually assumes XY is the main plane for 2D, but let's check.
        // Standard 3D view usually has Y up.
        // If CosmoSim 2D is X/Y, then we should rotate grid to be XY.
        // Let's assume standard orientation for now: Grid on XZ plane (y=0).
        // Wait, CosmoSim 2D is typically X/Y.
        // Let's rotate to align with XY plane.
        gridHelper.rotation.x = Math.PI / 2;
        this.group.add(gridHelper);

        // Add a border
        const geometry = new THREE.EdgesGeometry(new THREE.PlaneGeometry(size, size));
        const material = new THREE.LineBasicMaterial({ color: 0x666666 });
        const border = new THREE.LineSegments(geometry, material);
        this.group.add(border);
    }

    buildBubble(radius) {
        const geometry = new THREE.SphereGeometry(radius, 32, 24);
        const material = new THREE.MeshBasicMaterial({
            color: 0x5555ff,
            wireframe: true,
            transparent: true,
            opacity: 0.3
        });
        const mesh = new THREE.Mesh(geometry, material);
        this.group.add(mesh);
    }

    buildToroidal(bounds) {
        const size = bounds * 2;
        const geometry = new THREE.BoxGeometry(size, size, size);
        const edges = new THREE.EdgesGeometry(geometry);
        const material = new THREE.LineBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.5 });
        const box = new THREE.LineSegments(edges, material);
        this.group.add(box);
    }

    buildLattice2D(bounds) {
        const size = bounds * 2;
        const divisions = 40; // Denser than flat
        const gridHelper = new THREE.GridHelper(size, divisions, 0x008800, 0x004400);
        gridHelper.rotation.x = Math.PI / 2;
        this.group.add(gridHelper);
    }

    buildLattice3D(bounds) {
        const size = bounds * 2;
        const divisions = 4;
        const step = size / divisions;

        const material = new THREE.LineBasicMaterial({ color: 0x008800, transparent: true, opacity: 0.3 });
        const points = [];

        // Simple 3D grid lines
        for (let i = 0; i <= divisions; i++) {
            const p = -bounds + i * step;

            // X lines
            points.push(new THREE.Vector3(-bounds, p, -bounds));
            points.push(new THREE.Vector3(bounds, p, -bounds));
            points.push(new THREE.Vector3(-bounds, p, bounds));
            points.push(new THREE.Vector3(bounds, p, bounds));

            points.push(new THREE.Vector3(-bounds, -bounds, p));
            points.push(new THREE.Vector3(bounds, -bounds, p));
            points.push(new THREE.Vector3(-bounds, bounds, p));
            points.push(new THREE.Vector3(bounds, bounds, p));

            // Y lines
            points.push(new THREE.Vector3(p, -bounds, -bounds));
            points.push(new THREE.Vector3(p, bounds, -bounds));
            points.push(new THREE.Vector3(p, -bounds, bounds));
            points.push(new THREE.Vector3(p, bounds, bounds));

            // Z lines... (simplified, just doing a box for now + some internal lines)
        }

        // Use BoxHelper as a base
        const geometry = new THREE.BoxGeometry(size, size, size);
        const edges = new THREE.EdgesGeometry(geometry);
        const box = new THREE.LineSegments(edges, material);
        this.group.add(box);

        // Add a smaller internal box to hint at lattice
        const innerGeo = new THREE.BoxGeometry(size / 2, size / 2, size / 2);
        const innerEdges = new THREE.EdgesGeometry(innerGeo);
        const innerBox = new THREE.LineSegments(innerEdges, material);
        this.group.add(innerBox);
    }

    buildUnknown(bounds) {
        const size = bounds * 2;
        const geometry = new THREE.BoxGeometry(size, size, size);
        const edges = new THREE.EdgesGeometry(geometry);
        const material = new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.2 });
        const box = new THREE.LineSegments(edges, material);
        this.group.add(box);
    }
}
