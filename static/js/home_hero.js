(function () {
    window.__dataInsighterHeroInit = true;
    function cssVar(name, fallback) {
        const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
        return value || fallback;
    }

    function setThemeColors(target) {
        target.primary = new THREE.Color(cssVar('--primary-color', '#0f766e'));
        target.accent = new THREE.Color(cssVar('--accent-color', '#f97316'));
        target.info = new THREE.Color(cssVar('--info-color', '#2563eb'));
        target.text = new THREE.Color(cssVar('--text-color', '#152132'));
        target.soft = new THREE.Color(cssVar('--bg-soft', '#eef5f4'));
        target.surface = new THREE.Color(cssVar('--bg-canvas', '#ffffff'));
        target.grid = new THREE.Color(cssVar('--border-color', '#dbe5e4'));
    }

    function ready(fn) {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', fn, { once: true });
            return;
        }
        fn();
    }

    ready(function () {
        const mount = document.getElementById('hero3dScene');
        if (!mount || !window.THREE) {
            return;
        }

        const fallbackImage = mount.closest('.hero-visual-shell')?.querySelector('.hero-visual-runtime-fallback');

        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        const scene = new THREE.Scene();
        const colors = {};
        setThemeColors(colors);

        let renderer;
        try {
            renderer = new THREE.WebGLRenderer({
                antialias: true,
                alpha: true,
                powerPreference: 'high-performance',
            });
        } catch (error) {
            return;
        }
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.8));
        if ('outputColorSpace' in renderer && THREE.SRGBColorSpace) {
            renderer.outputColorSpace = THREE.SRGBColorSpace;
        } else if ('outputEncoding' in renderer && THREE.sRGBEncoding) {
            renderer.outputEncoding = THREE.sRGBEncoding;
        }
        renderer.setClearColor(0x000000, 0);
        mount.appendChild(renderer.domElement);
        if (fallbackImage) {
            fallbackImage.hidden = true;
        }

        const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
        camera.position.set(0, 0.4, 9.6);

        const root = new THREE.Group();
        root.position.set(0, 0.2, 0);
        scene.add(root);

        const ambient = new THREE.AmbientLight(0xffffff, 1.2);
        scene.add(ambient);

        const primaryLight = new THREE.PointLight(colors.primary, 18, 32, 2);
        primaryLight.position.set(-2.5, 2.2, 4.6);
        scene.add(primaryLight);

        const accentLight = new THREE.PointLight(colors.accent, 10, 24, 2);
        accentLight.position.set(3.8, -1.5, 4.2);
        scene.add(accentLight);

        const fillLight = new THREE.PointLight(colors.info, 9, 30, 2);
        fillLight.position.set(0.5, 3.6, -3.5);
        scene.add(fillLight);

        const floor = new THREE.Mesh(
            new THREE.CircleGeometry(5.6, 48),
            new THREE.MeshBasicMaterial({
                color: colors.soft,
                transparent: true,
                opacity: 0.12,
            })
        );
        floor.rotation.x = -Math.PI / 2;
        floor.position.set(0, -2.4, -0.2);
        root.add(floor);

        const grid = new THREE.GridHelper(10, 20, colors.grid, colors.grid);
        grid.material.transparent = true;
        grid.material.opacity = 0.18;
        grid.position.y = -2.3;
        root.add(grid);

        const coreGroup = new THREE.Group();
        root.add(coreGroup);

        const coreSphere = new THREE.Mesh(
            new THREE.SphereGeometry(1.02, 48, 48),
            new THREE.MeshPhysicalMaterial({
                color: colors.surface,
                roughness: 0.16,
                metalness: 0.05,
                transmission: 0.6,
                thickness: 1.2,
                transparent: true,
                opacity: 0.94,
                emissive: colors.primary.clone().multiplyScalar(0.2),
                clearcoat: 1,
                clearcoatRoughness: 0.2,
            })
        );
        coreGroup.add(coreSphere);

        const glow = new THREE.Mesh(
            new THREE.SphereGeometry(1.34, 48, 48),
            new THREE.MeshBasicMaterial({
                color: colors.primary,
                transparent: true,
                opacity: 0.12,
            })
        );
        coreGroup.add(glow);

        const ringPrimary = new THREE.Mesh(
            new THREE.TorusGeometry(1.76, 0.05, 20, 120),
            new THREE.MeshStandardMaterial({
                color: colors.primary,
                emissive: colors.primary,
                emissiveIntensity: 0.55,
                roughness: 0.3,
                metalness: 0.7,
            })
        );
        ringPrimary.rotation.x = Math.PI / 2.9;
        ringPrimary.rotation.y = Math.PI / 4.6;
        coreGroup.add(ringPrimary);

        const ringAccent = new THREE.Mesh(
            new THREE.TorusGeometry(1.38, 0.04, 18, 96),
            new THREE.MeshStandardMaterial({
                color: colors.accent,
                emissive: colors.accent,
                emissiveIntensity: 0.38,
                roughness: 0.36,
                metalness: 0.68,
            })
        );
        ringAccent.rotation.x = Math.PI / 2.1;
        ringAccent.rotation.z = Math.PI / 4.2;
        coreGroup.add(ringAccent);

        const nodes = new THREE.Group();
        for (let index = 0; index < 20; index += 1) {
            const node = new THREE.Mesh(
                new THREE.SphereGeometry(index % 3 === 0 ? 0.08 : 0.055, 18, 18),
                new THREE.MeshStandardMaterial({
                    color: index % 3 === 0 ? colors.info : colors.primary,
                    emissive: index % 3 === 0 ? colors.info : colors.primary,
                    emissiveIntensity: 0.6,
                    roughness: 0.28,
                    metalness: 0.2,
                })
            );
            const radius = 1.75 + (index % 5) * 0.13;
            const angle = (index / 20) * Math.PI * 2;
            const elevation = ((index % 4) - 1.5) * 0.42;
            node.position.set(
                Math.cos(angle) * radius,
                elevation,
                Math.sin(angle) * radius * 0.9
            );
            nodes.add(node);
        }
        root.add(nodes);

        const uploadCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(-4.2, 1.65, 0.2),
            new THREE.Vector3(-2.6, 1.2, 0.65),
            new THREE.Vector3(-1.45, 0.6, 0.4),
            new THREE.Vector3(-0.35, 0.05, 0.12),
        ]);

        const uploadLine = new THREE.Mesh(
            new THREE.TubeGeometry(uploadCurve, 80, 0.02, 10, false),
            new THREE.MeshBasicMaterial({
                color: colors.accent,
                transparent: true,
                opacity: 0.28,
            })
        );
        root.add(uploadLine);

        const uploadTokens = [];
        for (let index = 0; index < 4; index += 1) {
            const token = new THREE.Mesh(
                new THREE.BoxGeometry(0.62, 0.38, 0.14),
                new THREE.MeshPhysicalMaterial({
                    color: index % 2 === 0 ? colors.accent : colors.info,
                    emissive: index % 2 === 0 ? colors.accent : colors.info,
                    emissiveIntensity: 0.3,
                    roughness: 0.2,
                    metalness: 0.5,
                    clearcoat: 1,
                })
            );
            uploadTokens.push({ mesh: token, offset: index * 0.18 });
            root.add(token);
        }

        const outputCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(0.4, 0.05, -0.05),
            new THREE.Vector3(1.8, 0.45, 0.15),
            new THREE.Vector3(3.0, 0.7, -0.1),
            new THREE.Vector3(4.0, 1.0, 0.2),
        ]);

        const outputLine = new THREE.Mesh(
            new THREE.TubeGeometry(outputCurve, 80, 0.018, 10, false),
            new THREE.MeshBasicMaterial({
                color: colors.primary,
                transparent: true,
                opacity: 0.22,
            })
        );
        root.add(outputLine);

        const barsGroup = new THREE.Group();
        barsGroup.position.set(3.1, -1.1, -0.1);
        root.add(barsGroup);

        const barHeights = [1.15, 0.8, 1.5, 1.95, 1.35, 2.25];
        const bars = [];
        barHeights.forEach((height, index) => {
            const mesh = new THREE.Mesh(
                new THREE.BoxGeometry(0.28, height, 0.28),
                new THREE.MeshStandardMaterial({
                    color: index === barHeights.length - 1 ? colors.primary : colors.info,
                    emissive: index === barHeights.length - 1 ? colors.primary : colors.info,
                    emissiveIntensity: 0.22,
                    roughness: 0.24,
                    metalness: 0.52,
                })
            );
            mesh.position.set(index * 0.42 - 1.02, height / 2 - 0.4, 0);
            bars.push(mesh);
            barsGroup.add(mesh);
        });

        const panelMaterial = new THREE.MeshPhysicalMaterial({
            color: colors.surface,
            roughness: 0.12,
            metalness: 0.08,
            transparent: true,
            opacity: 0.92,
            transmission: 0.24,
            clearcoat: 1,
        });

        const dashboardPanel = new THREE.Mesh(
            new THREE.BoxGeometry(3.0, 1.7, 0.12),
            panelMaterial.clone()
        );
        dashboardPanel.position.set(2.8, 1.5, -1.0);
        dashboardPanel.rotation.set(-0.18, -0.52, -0.08);
        root.add(dashboardPanel);

        const reportPanel = new THREE.Mesh(
            new THREE.BoxGeometry(2.2, 1.45, 0.1),
            panelMaterial.clone()
        );
        reportPanel.position.set(3.55, -0.38, -1.55);
        reportPanel.rotation.set(0.12, -0.64, 0.08);
        root.add(reportPanel);

        const panelBars = new THREE.Group();
        for (let index = 0; index < 4; index += 1) {
            const strip = new THREE.Mesh(
                new THREE.BoxGeometry(0.18, 0.28 + index * 0.12, 0.02),
                new THREE.MeshStandardMaterial({
                    color: index % 2 === 0 ? colors.primary : colors.accent,
                    emissive: index % 2 === 0 ? colors.primary : colors.accent,
                    emissiveIntensity: 0.35,
                })
            );
            strip.position.set(-0.52 + index * 0.32, -0.12 + strip.geometry.parameters.height / 2, 0.08);
            dashboardPanel.add(strip);
            panelBars.add(strip);
        }

        const linePoints = [
            new THREE.Vector3(-0.95, 0.16, 0.09),
            new THREE.Vector3(-0.35, -0.1, 0.09),
            new THREE.Vector3(0.18, 0.22, 0.09),
            new THREE.Vector3(0.82, -0.04, 0.09),
        ];
        const dashboardLine = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(linePoints),
            new THREE.LineBasicMaterial({
                color: colors.info,
                transparent: true,
                opacity: 0.9,
            })
        );
        dashboardPanel.add(dashboardLine);

        const reportStrips = new THREE.Group();
        for (let index = 0; index < 3; index += 1) {
            const row = new THREE.Mesh(
                new THREE.BoxGeometry(1.35 - index * 0.1, 0.11, 0.02),
                new THREE.MeshBasicMaterial({
                    color: index === 0 ? colors.text : colors.grid,
                    transparent: true,
                    opacity: index === 0 ? 0.88 : 0.72,
                })
            );
            row.position.set(-0.12, 0.34 - index * 0.3, 0.08);
            reportPanel.add(row);
            reportStrips.add(row);
        }

        const orbitDots = new THREE.Points(
            new THREE.BufferGeometry(),
            new THREE.PointsMaterial({
                size: 0.06,
                transparent: true,
                opacity: 0.72,
                color: colors.primary,
            })
        );
        const orbitPositions = [];
        for (let index = 0; index < 80; index += 1) {
            const theta = Math.random() * Math.PI * 2;
            const radius = 1.5 + Math.random() * 1.8;
            orbitPositions.push(
                Math.cos(theta) * radius,
                (Math.random() - 0.5) * 2.6,
                Math.sin(theta) * radius
            );
        }
        orbitDots.geometry.setAttribute('position', new THREE.Float32BufferAttribute(orbitPositions, 3));
        root.add(orbitDots);

        const dashboardPoints = [];
        bars.forEach((bar) => {
            dashboardPoints.push(bar.position.clone().add(barsGroup.position));
        });
        const dashboardTrail = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(dashboardPoints),
            new THREE.LineBasicMaterial({
                color: colors.accent,
                transparent: true,
                opacity: 0.6,
            })
        );
        root.add(dashboardTrail);

        const pointer = { x: 0, y: 0 };
        mount.addEventListener('pointermove', function (event) {
            const rect = mount.getBoundingClientRect();
            pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            pointer.y = ((event.clientY - rect.top) / rect.height) * 2 - 1;
        });

        function resize() {
            const bounds = mount.getBoundingClientRect();
            renderer.setSize(bounds.width, bounds.height, false);
            camera.aspect = bounds.width / Math.max(bounds.height, 1);
            camera.updateProjectionMatrix();
        }

        resize();
        window.addEventListener('resize', resize);

        const clock = new THREE.Clock();
        let animationFrame = null;

        function applyTheme() {
            setThemeColors(colors);
            primaryLight.color.copy(colors.primary);
            accentLight.color.copy(colors.accent);
            fillLight.color.copy(colors.info);
            coreSphere.material.emissive.copy(colors.primary).multiplyScalar(0.2);
            coreSphere.material.color.copy(colors.surface);
            glow.material.color.copy(colors.primary);
            ringPrimary.material.color.copy(colors.primary);
            ringPrimary.material.emissive.copy(colors.primary);
            ringAccent.material.color.copy(colors.accent);
            ringAccent.material.emissive.copy(colors.accent);
            floor.material.color.copy(colors.soft);
            outputLine.material.color.copy(colors.primary);
            uploadLine.material.color.copy(colors.accent);
            dashboardPanel.material.color.copy(colors.surface);
            reportPanel.material.color.copy(colors.surface);
            dashboardLine.material.color.copy(colors.info);
            dashboardTrail.material.color.copy(colors.accent);
            orbitDots.material.color.copy(colors.primary);
            bars.forEach((bar, index) => {
                const tone = index === bars.length - 1 ? colors.primary : colors.info;
                bar.material.color.copy(tone);
                bar.material.emissive.copy(tone);
            });
            uploadTokens.forEach((token, index) => {
                const tone = index % 2 === 0 ? colors.accent : colors.info;
                token.mesh.material.color.copy(tone);
                token.mesh.material.emissive.copy(tone);
            });
            panelBars.children.forEach((strip, index) => {
                const tone = index % 2 === 0 ? colors.primary : colors.accent;
                strip.material.color.copy(tone);
                strip.material.emissive.copy(tone);
            });
            reportStrips.children.forEach((row, index) => {
                row.material.color.copy(index === 0 ? colors.text : colors.grid);
            });
            if (Array.isArray(grid.material)) {
                grid.material.forEach((material) => {
                    material.color.copy(colors.grid);
                    material.opacity = 0.18;
                });
            } else {
                grid.material.color.copy(colors.grid);
                grid.material.opacity = 0.18;
            }
        }

        window.addEventListener('di:theme-change', applyTheme);
        applyTheme();

        function render() {
            const elapsed = clock.getElapsedTime();
            const drift = prefersReducedMotion ? 0.2 : 1;

            root.rotation.y += ((pointer.x * 0.16) - root.rotation.y) * 0.04;
            root.rotation.x += (((-pointer.y * 0.08) - 0.02) - root.rotation.x) * 0.04;

            coreSphere.rotation.y = elapsed * 0.2 * drift;
            coreSphere.rotation.x = Math.sin(elapsed * 0.26) * 0.12;
            glow.scale.setScalar(1 + Math.sin(elapsed * 1.5) * 0.04);
            ringPrimary.rotation.z = elapsed * 0.24 * drift;
            ringPrimary.rotation.x = Math.PI / 2.9 + Math.sin(elapsed * 0.7) * 0.08;
            ringAccent.rotation.y = elapsed * -0.28 * drift;
            nodes.rotation.y = elapsed * 0.16 * drift;

            uploadTokens.forEach((token, index) => {
                const progress = (elapsed * 0.06 * drift + token.offset) % 1;
                const point = uploadCurve.getPointAt(progress);
                token.mesh.position.copy(point);
                token.mesh.rotation.x = 0.4 + Math.sin(elapsed + index) * 0.2;
                token.mesh.rotation.y = elapsed * 0.75 + index;
                const pulse = 0.92 + Math.sin(elapsed * 2.1 + index) * 0.08;
                token.mesh.scale.setScalar(pulse);
            });

            bars.forEach((bar, index) => {
                bar.scale.y = 0.94 + Math.sin(elapsed * 1.8 + index * 0.55) * 0.06;
            });

            dashboardPanel.position.y = 1.5 + Math.sin(elapsed * 0.9) * 0.12;
            reportPanel.position.y = -0.38 + Math.cos(elapsed * 0.8) * 0.1;
            orbitDots.rotation.y = elapsed * 0.06 * drift;
            orbitDots.rotation.x = Math.sin(elapsed * 0.08) * 0.06;

            renderer.render(scene, camera);
            animationFrame = window.requestAnimationFrame(render);
        }

        render();

        const observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (!entry.isIntersecting && animationFrame) {
                    window.cancelAnimationFrame(animationFrame);
                    animationFrame = null;
                } else if (entry.isIntersecting && !animationFrame) {
                    clock.start();
                    render();
                }
            });
        }, { threshold: 0.05 });

        observer.observe(mount);
    });
})();
