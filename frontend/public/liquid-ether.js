/* LiquidEther — fluid sim adapted for AURAI React SPA
   Differences from waitlist version:
   - initLiquidEther(containerId) accepts a container element ID
   - No auto-bootstrap; window.initLiquidEther must be called explicitly
   - window._leDestroy() stops the loop and removes the WebGL canvas
*/
(function () {
  'use strict';

  function initLiquidEther(containerId) {
    var container = document.getElementById(containerId || 'liquid-bg');
    if (!container || typeof THREE === 'undefined') return;

    // ── Palette & background colour ──────────────────────────
    var COLORS   = ['#5227FF', '#FF9FFC', '#B497CF'];
    var BG_VEC4  = new THREE.Vector4(0.039, 0.082, 0.188, 1.0); // #0a1530

    function makePaletteTex(stops) {
      var arr = stops.length === 1 ? [stops[0], stops[0]] : stops;
      var w   = arr.length;
      var data = new Uint8Array(w * 4);
      for (var i = 0; i < w; i++) {
        var c = new THREE.Color(arr[i]);
        data[i*4]   = Math.round(c.r * 255);
        data[i*4+1] = Math.round(c.g * 255);
        data[i*4+2] = Math.round(c.b * 255);
        data[i*4+3] = 255;
      }
      var tex = new THREE.DataTexture(data, w, 1, THREE.RGBAFormat);
      tex.magFilter = THREE.LinearFilter;
      tex.minFilter = THREE.LinearFilter;
      tex.wrapS     = THREE.ClampToEdgeWrapping;
      tex.wrapT     = THREE.ClampToEdgeWrapping;
      tex.generateMipmaps = false;
      tex.needsUpdate = true;
      return tex;
    }
    var paletteTex = makePaletteTex(COLORS);

    // ── Sim config ───────────────────────────────────────────
    var CFG = {
      mouseForce:        20,
      cursorSize:        100,
      isViscous:         true,
      viscous:           30,
      iterationsViscous: 32,
      iterationsPoisson: 32,
      dt:                0.014,
      BFECC:             true,
      resolution:        0.5,
      isBounce:          false,
      autoDemo:          true,
      autoSpeed:         0.5,
      autoIntensity:     2.2,
      takeoverDuration:  0.25,
      autoResumeDelay:   800,
      autoRampDuration:  0.6
    };

    // ── Common ───────────────────────────────────────────────
    var Common = {
      width: 1, height: 1, pixelRatio: 1, time: 0, delta: 0,
      container: null, renderer: null, clock: null,
      init: function (c) {
        this.container  = c;
        this.pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        this.resize();
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.autoClear = false;
        this.renderer.setClearColor(new THREE.Color(0x000000), 0);
        this.renderer.setPixelRatio(this.pixelRatio);
        this.renderer.setSize(this.width, this.height);
        this.renderer.domElement.style.cssText = 'width:100%;height:100%;display:block;';
        this.clock = new THREE.Clock();
        this.clock.start();
      },
      resize: function () {
        if (!this.container) return;
        var r = this.container.getBoundingClientRect();
        this.width  = Math.max(1, Math.floor(r.width));
        this.height = Math.max(1, Math.floor(r.height));
        if (this.renderer) this.renderer.setSize(this.width, this.height, false);
      },
      update: function () {
        this.delta = this.clock.getDelta();
        this.time += this.delta;
      }
    };

    // ── Mouse ────────────────────────────────────────────────
    var Mouse = {
      mouseMoved: false,
      coords:    new THREE.Vector2(),
      coords_old: new THREE.Vector2(),
      diff:      new THREE.Vector2(),
      timer:     null,
      container: null,
      listenerTarget: null,
      docTarget: null,
      isHoverInside:  false,
      hasUserControl: false,
      isAutoActive:   false,
      autoIntensity:  2.0,
      takeoverActive: false,
      takeoverStartTime: 0,
      takeoverDuration:  0.25,
      takeoverFrom: new THREE.Vector2(),
      takeoverTo:   new THREE.Vector2(),
      onInteract: null,

      init: function (c) {
        this.container  = c;
        this.docTarget  = c.ownerDocument || document;
        var win = this.docTarget.defaultView || window;
        this.listenerTarget = win;
        var self = this;
        this._mm  = function (e) { self.onMouseMove(e); };
        this._ts  = function (e) { self.onTouchStart(e); };
        this._tm  = function (e) { self.onTouchMove(e); };
        this._te  = function ()  { self.isHoverInside = false; };
        this._dl  = function ()  { self.isHoverInside = false; };
        win.addEventListener('mousemove',  this._mm);
        win.addEventListener('touchstart', this._ts, { passive: true });
        win.addEventListener('touchmove',  this._tm, { passive: true });
        win.addEventListener('touchend',   this._te);
        this.docTarget.addEventListener('mouseleave', this._dl);
      },

      destroy: function () {
        var win = this.listenerTarget;
        if (!win) return;
        win.removeEventListener('mousemove',  this._mm);
        win.removeEventListener('touchstart', this._ts);
        win.removeEventListener('touchmove',  this._tm);
        win.removeEventListener('touchend',   this._te);
        if (this.docTarget) this.docTarget.removeEventListener('mouseleave', this._dl);
      },

      isPointInside: function (cx, cy) {
        if (!this.container) return false;
        var r = this.container.getBoundingClientRect();
        return cx >= r.left && cx <= r.right && cy >= r.top && cy <= r.bottom;
      },

      setCoords: function (x, y) {
        if (!this.container) return;
        if (this.timer) clearTimeout(this.timer);
        var r = this.container.getBoundingClientRect();
        if (!r.width || !r.height) return;
        var nx = (x - r.left) / r.width;
        var ny = (y - r.top)  / r.height;
        this.coords.set(nx * 2 - 1, -(ny * 2 - 1));
        this.mouseMoved = true;
        var self = this;
        this.timer = setTimeout(function () { self.mouseMoved = false; }, 100);
      },

      setNormalized: function (nx, ny) {
        this.coords.set(nx, ny);
        this.mouseMoved = true;
      },

      onMouseMove: function (e) {
        if (!this.isPointInside(e.clientX, e.clientY)) { this.isHoverInside = false; return; }
        this.isHoverInside = true;
        if (this.onInteract) this.onInteract();
        if (this.isAutoActive && !this.hasUserControl && !this.takeoverActive) {
          var r = this.container.getBoundingClientRect();
          if (!r.width || !r.height) return;
          var nx = (e.clientX - r.left) / r.width;
          var ny = (e.clientY - r.top)  / r.height;
          this.takeoverFrom.copy(this.coords);
          this.takeoverTo.set(nx * 2 - 1, -(ny * 2 - 1));
          this.takeoverStartTime = performance.now();
          this.takeoverActive  = true;
          this.hasUserControl  = true;
          this.isAutoActive    = false;
          return;
        }
        this.setCoords(e.clientX, e.clientY);
        this.hasUserControl = true;
      },

      // Touch input is intentionally ignored. On a content page every touch
      // is a scroll gesture; routing it into the fluid sim made the gradient
      // chase the finger down the page and overwrite the auto-driver's ripples.
      onTouchStart: function (_e) { /* no-op on touch devices */ },
      onTouchMove:  function (_e) { /* no-op on touch devices */ },

      update: function () {
        if (this.takeoverActive) {
          var t = (performance.now() - this.takeoverStartTime) / (this.takeoverDuration * 1000);
          if (t >= 1) {
            this.takeoverActive = false;
            this.coords.copy(this.takeoverTo);
            this.coords_old.copy(this.coords);
            this.diff.set(0, 0);
          } else {
            var k = t * t * (3 - 2 * t);
            this.coords.copy(this.takeoverFrom).lerp(this.takeoverTo, k);
          }
        }
        this.diff.subVectors(this.coords, this.coords_old);
        this.coords_old.copy(this.coords);
        if (this.coords_old.x === 0 && this.coords_old.y === 0) this.diff.set(0, 0);
        if (this.isAutoActive && !this.takeoverActive) this.diff.multiplyScalar(this.autoIntensity);
      }
    };

    // ── AutoDriver ───────────────────────────────────────────
    function AutoDriver(mouse, getLastInteraction, opts) {
      this.mouse   = mouse;
      this.getLast = getLastInteraction;
      this.enabled = opts.enabled;
      this.speed   = opts.speed;
      this.resumeDelay    = opts.resumeDelay || 3000;
      this.rampDurationMs = (opts.rampDuration || 0) * 1000;
      this.active  = false;
      this.current = new THREE.Vector2(0, 0);
      this.target  = new THREE.Vector2();
      this.lastTime       = performance.now();
      this.activationTime = 0;
      this.margin  = 0.2;
      this._dir    = new THREE.Vector2();
      this.pickTarget();
    }
    AutoDriver.prototype.pickTarget = function () {
      var m = 1 - this.margin;
      this.target.set((Math.random() * 2 - 1) * m, (Math.random() * 2 - 1) * m);
    };
    AutoDriver.prototype.forceStop = function () {
      this.active = false;
      this.mouse.isAutoActive = false;
    };
    AutoDriver.prototype.update = function () {
      if (!this.enabled) return;
      var now  = performance.now();
      var idle = now - this.getLast();
      if (idle < this.resumeDelay) { if (this.active) this.forceStop(); return; }
      if (this.mouse.isHoverInside) { if (this.active) this.forceStop(); return; }
      if (!this.active) {
        this.active = true;
        this.current.copy(this.mouse.coords);
        this.lastTime       = now;
        this.activationTime = now;
      }
      this.mouse.isAutoActive = true;
      var dt = Math.min((now - this.lastTime) / 1000, 0.2);
      this.lastTime = now;
      var dir  = this._dir.subVectors(this.target, this.current);
      var dist = dir.length();
      if (dist < 0.01) { this.pickTarget(); return; }
      dir.normalize();
      var ramp = 1;
      if (this.rampDurationMs > 0) {
        var tr = Math.min(1, (now - this.activationTime) / this.rampDurationMs);
        ramp = tr * tr * (3 - 2 * tr);
      }
      this.current.addScaledVector(dir, Math.min(this.speed * dt * ramp, dist));
      this.mouse.setNormalized(this.current.x, this.current.y);
    };

    // ── GLSL shaders ─────────────────────────────────────────
    var face_vert = [
      'attribute vec3 position;',
      'uniform vec2 px;',
      'uniform vec2 boundarySpace;',
      'varying vec2 uv;',
      'precision highp float;',
      'void main(){',
      '  vec3 pos = position;',
      '  vec2 scale = 1.0 - boundarySpace * 2.0;',
      '  pos.xy = pos.xy * scale;',
      '  uv = vec2(0.5) + pos.xy * 0.5;',
      '  gl_Position = vec4(pos, 1.0);',
      '}'
    ].join('\n');

    var line_vert = [
      'attribute vec3 position;',
      'uniform vec2 px;',
      'precision highp float;',
      'varying vec2 uv;',
      'void main(){',
      '  vec3 pos = position;',
      '  uv = 0.5 + pos.xy * 0.5;',
      '  vec2 n = sign(pos.xy);',
      '  pos.xy = abs(pos.xy) - px * 1.0;',
      '  pos.xy *= n;',
      '  gl_Position = vec4(pos, 1.0);',
      '}'
    ].join('\n');

    var mouse_vert = [
      'precision highp float;',
      'attribute vec3 position;',
      'attribute vec2 uv;',
      'uniform vec2 center;',
      'uniform vec2 scale;',
      'uniform vec2 px;',
      'varying vec2 vUv;',
      'void main(){',
      '  vec2 pos = position.xy * scale * 2.0 * px + center;',
      '  vUv = uv;',
      '  gl_Position = vec4(pos, 0.0, 1.0);',
      '}'
    ].join('\n');

    var advection_frag = [
      'precision highp float;',
      'uniform sampler2D velocity;',
      'uniform float dt;',
      'uniform bool isBFECC;',
      'uniform vec2 fboSize;',
      'uniform vec2 px;',
      'varying vec2 uv;',
      'void main(){',
      '  vec2 ratio = max(fboSize.x, fboSize.y) / fboSize;',
      '  if(isBFECC == false){',
      '    vec2 vel = texture2D(velocity, uv).xy;',
      '    vec2 uv2 = uv - vel * dt * ratio;',
      '    gl_FragColor = vec4(texture2D(velocity, uv2).xy, 0.0, 0.0);',
      '  } else {',
      '    vec2 spot_new  = uv;',
      '    vec2 vel_old   = texture2D(velocity, uv).xy;',
      '    vec2 spot_old  = spot_new - vel_old * dt * ratio;',
      '    vec2 vel_new1  = texture2D(velocity, spot_old).xy;',
      '    vec2 spot_new2 = spot_old + vel_new1 * dt * ratio;',
      '    vec2 error     = spot_new2 - spot_new;',
      '    vec2 spot_new3 = spot_new - error / 2.0;',
      '    vec2 vel_2     = texture2D(velocity, spot_new3).xy;',
      '    vec2 spot_old2 = spot_new3 - vel_2 * dt * ratio;',
      '    gl_FragColor   = vec4(texture2D(velocity, spot_old2).xy, 0.0, 0.0);',
      '  }',
      '}'
    ].join('\n');

    var color_frag = [
      'precision highp float;',
      'uniform sampler2D velocity;',
      'uniform sampler2D palette;',
      'uniform vec4 bgColor;',
      'varying vec2 uv;',
      'void main(){',
      '  vec2 vel  = texture2D(velocity, uv).xy;',
      '  float lv  = clamp(length(vel), 0.0, 1.0);',
      '  vec3 c    = texture2D(palette, vec2(lv, 0.5)).rgb;',
      '  vec3 rgb  = mix(bgColor.rgb, c, lv);',
      '  float a   = mix(bgColor.a,   1.0, lv);',
      '  gl_FragColor = vec4(rgb, a);',
      '}'
    ].join('\n');

    var divergence_frag = [
      'precision highp float;',
      'uniform sampler2D velocity;',
      'uniform float dt;',
      'uniform vec2 px;',
      'varying vec2 uv;',
      'void main(){',
      '  float x0 = texture2D(velocity, uv - vec2(px.x, 0.0)).x;',
      '  float x1 = texture2D(velocity, uv + vec2(px.x, 0.0)).x;',
      '  float y0 = texture2D(velocity, uv - vec2(0.0, px.y)).y;',
      '  float y1 = texture2D(velocity, uv + vec2(0.0, px.y)).y;',
      '  gl_FragColor = vec4((x1 - x0 + y1 - y0) * 0.5 / dt);',
      '}'
    ].join('\n');

    var externalForce_frag = [
      'precision highp float;',
      'uniform vec2 force;',
      'uniform vec2 center;',
      'uniform vec2 scale;',
      'uniform vec2 px;',
      'varying vec2 vUv;',
      'void main(){',
      '  vec2 circle = (vUv - 0.5) * 2.0;',
      '  float d = 1.0 - min(length(circle), 1.0);',
      '  d *= d;',
      '  gl_FragColor = vec4(force * d, 0.0, 1.0);',
      '}'
    ].join('\n');

    var poisson_frag = [
      'precision highp float;',
      'uniform sampler2D pressure;',
      'uniform sampler2D divergence;',
      'uniform vec2 px;',
      'varying vec2 uv;',
      'void main(){',
      '  float p0 = texture2D(pressure, uv + vec2(px.x * 2.0, 0.0)).r;',
      '  float p1 = texture2D(pressure, uv - vec2(px.x * 2.0, 0.0)).r;',
      '  float p2 = texture2D(pressure, uv + vec2(0.0, px.y * 2.0)).r;',
      '  float p3 = texture2D(pressure, uv - vec2(0.0, px.y * 2.0)).r;',
      '  float div = texture2D(divergence, uv).r;',
      '  gl_FragColor = vec4((p0 + p1 + p2 + p3) / 4.0 - div);',
      '}'
    ].join('\n');

    var pressure_frag = [
      'precision highp float;',
      'uniform sampler2D pressure;',
      'uniform sampler2D velocity;',
      'uniform vec2 px;',
      'uniform float dt;',
      'varying vec2 uv;',
      'void main(){',
      '  float p0 = texture2D(pressure, uv + vec2(px.x, 0.0)).r;',
      '  float p1 = texture2D(pressure, uv - vec2(px.x, 0.0)).r;',
      '  float p2 = texture2D(pressure, uv + vec2(0.0, px.y)).r;',
      '  float p3 = texture2D(pressure, uv - vec2(0.0, px.y)).r;',
      '  vec2 v = texture2D(velocity, uv).xy;',
      '  v -= vec2(p0 - p1, p2 - p3) * 0.5 * dt;',
      '  gl_FragColor = vec4(v, 0.0, 1.0);',
      '}'
    ].join('\n');

    var viscous_frag = [
      'precision highp float;',
      'uniform sampler2D velocity;',
      'uniform sampler2D velocity_new;',
      'uniform float v;',
      'uniform vec2 px;',
      'uniform float dt;',
      'varying vec2 uv;',
      'void main(){',
      '  vec2 old  = texture2D(velocity, uv).xy;',
      '  vec2 n0   = texture2D(velocity_new, uv + vec2(px.x*2.0, 0.0)).xy;',
      '  vec2 n1   = texture2D(velocity_new, uv - vec2(px.x*2.0, 0.0)).xy;',
      '  vec2 n2   = texture2D(velocity_new, uv + vec2(0.0, px.y*2.0)).xy;',
      '  vec2 n3   = texture2D(velocity_new, uv - vec2(0.0, px.y*2.0)).xy;',
      '  vec2 newv = 4.0*old + v*dt*(n0+n1+n2+n3);',
      '  gl_FragColor = vec4(newv / (4.0*(1.0 + v*dt)), 0.0, 0.0);',
      '}'
    ].join('\n');

    // ── FBO helper ───────────────────────────────────────────
    function makeFBO(w, h) {
      var type = /iPad|iPhone|iPod/i.test(navigator.userAgent) ? THREE.HalfFloatType : THREE.FloatType;
      return new THREE.WebGLRenderTarget(w, h, {
        type: type, depthBuffer: false, stencilBuffer: false,
        minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
        wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping
      });
    }

    function makeScene() {
      return { scene: new THREE.Scene(), camera: new THREE.Camera() };
    }

    function renderPass(scene, camera, output) {
      Common.renderer.setRenderTarget(output || null);
      Common.renderer.render(scene, camera);
      Common.renderer.setRenderTarget(null);
    }

    // ── Simulation ───────────────────────────────────────────
    function Simulation() {
      this.fboSize       = new THREE.Vector2();
      this.cellScale     = new THREE.Vector2();
      this.boundarySpace = new THREE.Vector2();
      this.fbos = {};
      this._calcSize();
      this._createFBOs();
      this._createPasses();
    }

    Simulation.prototype._calcSize = function () {
      var w = Math.max(1, Math.round(CFG.resolution * Common.width));
      var h = Math.max(1, Math.round(CFG.resolution * Common.height));
      this.cellScale.set(1 / w, 1 / h);
      this.fboSize.set(w, h);
    };

    Simulation.prototype._createFBOs = function () {
      var names = ['vel_0','vel_1','vel_v0','vel_v1','div','p0','p1'];
      for (var i = 0; i < names.length; i++) {
        this.fbos[names[i]] = makeFBO(this.fboSize.x, this.fboSize.y);
      }
    };

    Simulation.prototype._createPasses = function () {
      var cs = this.cellScale, fs = this.fboSize, bs = this.boundarySpace;
      var f  = this.fbos;

      var advUni = {
        boundarySpace: { value: cs }, px: { value: cs },
        fboSize: { value: fs }, velocity: { value: f.vel_0.texture },
        dt: { value: CFG.dt }, isBFECC: { value: true }
      };
      var advS = makeScene();
      advS.mat = new THREE.RawShaderMaterial({ vertexShader: face_vert, fragmentShader: advection_frag, uniforms: advUni });
      advS.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2), advS.mat));
      var bg = new THREE.BufferGeometry();
      bg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(
        [-1,-1,0, -1,1,0, -1,1,0, 1,1,0, 1,1,0, 1,-1,0, 1,-1,0, -1,-1,0]
      ), 3));
      advS.line = new THREE.LineSegments(bg, new THREE.RawShaderMaterial({
        vertexShader: line_vert, fragmentShader: advection_frag, uniforms: advUni
      }));
      advS.scene.add(advS.line);
      this.adv = advS; this.adv.uni = advUni; this.adv.output = f.vel_1;

      var forceUni = {
        px: { value: cs }, force: { value: new THREE.Vector2() },
        center: { value: new THREE.Vector2() },
        scale:  { value: new THREE.Vector2(CFG.cursorSize, CFG.cursorSize) }
      };
      var frcS = makeScene();
      frcS.mesh = new THREE.Mesh(
        new THREE.PlaneGeometry(1,1),
        new THREE.RawShaderMaterial({
          vertexShader: mouse_vert, fragmentShader: externalForce_frag,
          blending: THREE.AdditiveBlending, depthWrite: false, uniforms: forceUni
        })
      );
      frcS.scene.add(frcS.mesh);
      this.frc = frcS; this.frc.uni = forceUni; this.frc.output = f.vel_1;

      var visUni = {
        boundarySpace: { value: bs }, velocity: { value: f.vel_1.texture },
        velocity_new: { value: f.vel_v0.texture }, v: { value: CFG.viscous },
        px: { value: cs }, dt: { value: CFG.dt }
      };
      var visS = makeScene();
      visS.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),
        new THREE.RawShaderMaterial({ vertexShader: face_vert, fragmentShader: viscous_frag, uniforms: visUni })
      ));
      this.vis = visS; this.vis.uni = visUni;
      this.vis.out0 = f.vel_v0; this.vis.out1 = f.vel_v1;

      var divUni = {
        boundarySpace: { value: bs }, velocity: { value: f.vel_v0.texture },
        px: { value: cs }, dt: { value: CFG.dt }
      };
      var divS = makeScene();
      divS.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),
        new THREE.RawShaderMaterial({ vertexShader: face_vert, fragmentShader: divergence_frag, uniforms: divUni })
      ));
      this.div = divS; this.div.uni = divUni; this.div.output = f.div;

      var psnUni = {
        boundarySpace: { value: bs }, pressure: { value: f.p0.texture },
        divergence: { value: f.div.texture }, px: { value: cs }
      };
      var psnS = makeScene();
      psnS.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),
        new THREE.RawShaderMaterial({ vertexShader: face_vert, fragmentShader: poisson_frag, uniforms: psnUni })
      ));
      this.psn = psnS; this.psn.uni = psnUni;
      this.psn.out0 = f.p0; this.psn.out1 = f.p1;

      var preUni = {
        boundarySpace: { value: bs }, pressure: { value: f.p0.texture },
        velocity: { value: f.vel_v0.texture }, px: { value: cs }, dt: { value: CFG.dt }
      };
      var preS = makeScene();
      preS.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),
        new THREE.RawShaderMaterial({ vertexShader: face_vert, fragmentShader: pressure_frag, uniforms: preUni })
      ));
      this.pre = preS; this.pre.uni = preUni; this.pre.output = f.vel_0;
    };

    Simulation.prototype.resize = function () {
      this._calcSize();
      for (var k in this.fbos) this.fbos[k].setSize(this.fboSize.x, this.fboSize.y);
    };

    Simulation.prototype.update = function () {
      var f   = this.fbos;
      var bs  = this.boundarySpace;
      var cs  = this.cellScale;
      bs.copy(CFG.isBounce ? new THREE.Vector2() : cs);

      this.adv.uni.dt.value     = CFG.dt;
      this.adv.uni.isBFECC.value = CFG.BFECC;
      this.adv.line.visible     = CFG.isBounce;
      renderPass(this.adv.scene, this.adv.camera, this.adv.output);

      var fx = (Mouse.diff.x / 2) * CFG.mouseForce;
      var fy = (Mouse.diff.y / 2) * CFG.mouseForce;
      var sx = CFG.cursorSize * cs.x;
      var sy = CFG.cursorSize * cs.y;
      this.frc.uni.force.value.set(fx, fy);
      this.frc.uni.center.value.set(
        Math.min(Math.max(Mouse.coords.x, -1+sx+cs.x*2), 1-sx-cs.x*2),
        Math.min(Math.max(Mouse.coords.y, -1+sy+cs.y*2), 1-sy-cs.y*2)
      );
      this.frc.uni.scale.value.set(CFG.cursorSize, CFG.cursorSize);
      renderPass(this.frc.scene, this.frc.camera, this.frc.output);

      var vel = f.vel_1;
      if (CFG.isViscous) {
        this.vis.uni.v.value  = CFG.viscous;
        this.vis.uni.dt.value = CFG.dt;
        var vIn, vOut;
        for (var vi = 0; vi < CFG.iterationsViscous; vi++) {
          if (vi % 2 === 0) { vIn = this.vis.out0; vOut = this.vis.out1; }
          else              { vIn = this.vis.out1; vOut = this.vis.out0; }
          this.vis.uni.velocity_new.value = vIn.texture;
          renderPass(this.vis.scene, this.vis.camera, vOut);
        }
        vel = vOut;
      }

      this.div.uni.velocity.value = vel.texture;
      renderPass(this.div.scene, this.div.camera, this.div.output);

      var pIn, pOut;
      for (var pi = 0; pi < CFG.iterationsPoisson; pi++) {
        if (pi % 2 === 0) { pIn = this.psn.out0; pOut = this.psn.out1; }
        else              { pIn = this.psn.out1; pOut = this.psn.out0; }
        this.psn.uni.pressure.value = pIn.texture;
        renderPass(this.psn.scene, this.psn.camera, pOut);
      }
      var pressure = pOut;

      this.pre.uni.velocity.value = vel.texture;
      this.pre.uni.pressure.value = pressure.texture;
      renderPass(this.pre.scene, this.pre.camera, this.pre.output);
    };

    // ── Output (colour pass) ─────────────────────────────────
    Common.init(container);
    Mouse.init(container);
    Mouse.autoIntensity    = CFG.autoIntensity;
    Mouse.takeoverDuration = CFG.takeoverDuration;

    var lastInteraction = performance.now();
    Mouse.onInteract = function () { lastInteraction = performance.now(); if (driver) driver.forceStop(); };

    var sim = new Simulation();

    var outScene  = new THREE.Scene();
    var outCamera = new THREE.Camera();
    var outMesh   = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.RawShaderMaterial({
        vertexShader: face_vert, fragmentShader: color_frag,
        transparent: true, depthWrite: false,
        uniforms: {
          velocity:      { value: sim.fbos.vel_0.texture },
          boundarySpace: { value: new THREE.Vector2() },
          palette:       { value: paletteTex },
          bgColor:       { value: BG_VEC4 }
        }
      })
    );
    outScene.add(outMesh);

    container.prepend(Common.renderer.domElement);

    var driver = new AutoDriver(Mouse, function () { return lastInteraction; }, {
      enabled: CFG.autoDemo, speed: CFG.autoSpeed,
      resumeDelay: CFG.autoResumeDelay, rampDuration: CFG.autoRampDuration
    });

    // ── Render loop ──────────────────────────────────────────
    var rafId = null;
    var running = false;

    function loop() {
      if (!running) return;
      driver.update();
      Mouse.update();
      Common.update();
      sim.update();
      Common.renderer.setRenderTarget(null);
      Common.renderer.render(outScene, outCamera);
      rafId = requestAnimationFrame(loop);
    }

    function start() { if (running) return; running = true; loop(); }
    function pause() { running = false; if (rafId) { cancelAnimationFrame(rafId); rafId = null; } }

    // Guard sim.resize() behind a width-change check. iOS Safari fires `resize`
    // continuously during scroll as the URL bar animates (height-only changes).
    // sim.resize() calls setSize() on every fluid FBO, wiping velocity textures.
    // We only re-size the sim when WIDTH actually changes (orientation flip,
    // desktop resize) — height-only deltas leave the sim slightly stretched,
    // which is invisible for a decorative background.
    var lastSimWidth = Common.width;
    function onResize() {
      Common.resize();
      if (Math.abs(Common.width - lastSimWidth) > 2) {
        lastSimWidth = Common.width;
        sim.resize();
      }
    }
    window.addEventListener('resize', onResize);

    function onVisibility() {
      if (document.hidden) pause(); else start();
    }
    document.addEventListener('visibilitychange', onVisibility);

    // On mobile, RAF is throttled during scroll and IntersectionObserver on a
    // fixed element can fire isIntersecting=false mid-scroll, permanently pausing
    // the loop. Guard against that by restarting on touchend.
    function onTouchEnd() {
      if (!document.hidden && running === false) start();
    }
    window.addEventListener('touchend', onTouchEnd, { passive: true });

    // ── Expose cleanup for React SPA unmount ─────────────────
    window._leDestroy = function () {
      pause();
      window.removeEventListener('resize', onResize);
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('touchend', onTouchEnd);
      Mouse.destroy();
      if (Common.renderer) {
        Common.renderer.domElement.remove();
        Common.renderer.dispose();
        Common.renderer = null;
      }
      window._leDestroy = null;
    };

    start();
  }

  // ── Expose — React component calls this on mount ─────────
  window.initLiquidEther = initLiquidEther;
})();
