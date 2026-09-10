import { useEffect, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, Html, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { Cloud, Lane, TrafficLight, WorldState } from "../types";

/** ego (x fwd, y left, z up) -> three (x right, y up, z toward viewer) */
const toThree = (p: [number, number, number] | number[]): [number, number, number] => [-p[1], p[2], -p[0]];

const LAMP_COLORS: Record<string, string> = { red: "#ff3b30", yellow: "#ffcc00", green: "#34c759", empty: "#666" };

// All geometry below is preallocated once and updated in place: creating new BufferAttributes every frame
// leaks GPU buffers (three.js never frees replaced attributes) and eventually kills the tab.

const CLOUD_CAPACITY = 80000;

function PointCloud({ cloud }: { cloud: Cloud | null }) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(CLOUD_CAPACITY * 3), 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(CLOUD_CAPACITY * 3), 3).setUsage(THREE.DynamicDrawUsage));
    g.setDrawRange(0, 0);
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, -20), 200);
    return g;
  }, []);
  useEffect(() => () => geom.dispose(), [geom]);
  useEffect(() => {
    if (!cloud) return;
    const n = Math.min(cloud.n, CLOUD_CAPACITY);
    const pos = geom.getAttribute("position") as THREE.BufferAttribute;
    const col = geom.getAttribute("color") as THREE.BufferAttribute;
    const pa = pos.array as Float32Array, ca = col.array as Float32Array;
    for (let i = 0; i < n; i++) {
      pa[i * 3] = -cloud.xyz[i * 3 + 1];
      pa[i * 3 + 1] = cloud.xyz[i * 3 + 2];
      pa[i * 3 + 2] = -cloud.xyz[i * 3];
      ca[i * 3] = cloud.rgb[i * 3] / 255;
      ca[i * 3 + 1] = cloud.rgb[i * 3 + 1] / 255;
      ca[i * 3 + 2] = cloud.rgb[i * 3 + 2] / 255;
    }
    pos.addUpdateRange(0, n * 3);
    col.addUpdateRange(0, n * 3);
    pos.needsUpdate = true;
    col.needsUpdate = true;
    geom.setDrawRange(0, n);
  }, [cloud, geom]);
  return (
    <points geometry={geom} frustumCulled={false}>
      <pointsMaterial size={0.12} vertexColors sizeAttenuation />
    </points>
  );
}

/** Flat ribbon along a polyline (width in metres), fixed capacity, updated in place. */
const RIBBON_CAPACITY = 64;

function Ribbon({ points, color, width, opacity }: { points: [number, number, number][]; color: string; width: number; opacity: number }) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(RIBBON_CAPACITY * 2 * 3), 3).setUsage(THREE.DynamicDrawUsage));
    const idx = new Uint16Array((RIBBON_CAPACITY - 1) * 6);
    for (let i = 0; i < RIBBON_CAPACITY - 1; i++) {
      const a = i * 2, b = a + 1, c = a + 2, d = a + 3;
      idx.set([a, b, c, b, d, c], i * 6);
    }
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.setDrawRange(0, 0);
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, -20), 100);
    return g;
  }, []);
  useEffect(() => () => geom.dispose(), [geom]);
  useEffect(() => {
    const n = Math.min(points.length, RIBBON_CAPACITY);
    const pos = geom.getAttribute("position") as THREE.BufferAttribute;
    const a = pos.array as Float32Array;
    for (let i = 0; i < n; i++) {
      const p = points[i];
      const q = points[Math.min(i + 1, n - 1)], r = points[Math.max(i - 1, 0)];
      // direction in the ground plane (three x/z), perpendicular offset
      let dx = q[0] - r[0], dz = q[2] - r[2];
      const len = Math.hypot(dx, dz) || 1;
      dx /= len; dz /= len;
      const ox = -dz * width / 2, oz = dx * width / 2;
      a[i * 6] = p[0] + ox; a[i * 6 + 1] = p[1] + 0.02; a[i * 6 + 2] = p[2] + oz;
      a[i * 6 + 3] = p[0] - ox; a[i * 6 + 4] = p[1] + 0.02; a[i * 6 + 5] = p[2] - oz;
    }
    pos.addUpdateRange(0, n * 6);
    pos.needsUpdate = true;
    geom.setDrawRange(0, Math.max(0, (n - 1) * 6));
  }, [points, geom, width]);
  return (
    <mesh geometry={geom} frustumCulled={false}>
      <meshBasicMaterial color={color} transparent opacity={opacity} side={THREE.DoubleSide} depthWrite={false} />
    </mesh>
  );
}

function LaneItem({ lane }: { lane: Lane }) {
  const pts = useMemo(() => lane.points_3d.map(toThree), [lane.points_3d]);
  const ego = lane.role.startsWith("ego");
  const color = lane.coasting ? "#9a9a9a" : ego ? "#34c759" : "#ffb340";
  return <Ribbon points={pts} color={color} width={ego ? 0.3 : 0.18} opacity={0.35 + 0.65 * Math.min(1, lane.confidence)} />;
}

function LightItem({ lt, associated }: { lt: TrafficLight; associated: boolean }) {
  const p = toThree(lt.position_ego);
  const main = lt.lamps.find((l) => l.shape === "circle") ?? lt.lamps[0];
  const color = lt.coasting ? "#777" : main ? LAMP_COLORS[main.color] ?? "#fff" : "#fff";
  const alpha = 0.3 + 0.7 * Math.min(1, lt.distance_confidence);
  return (
    <group>
      <mesh position={[p[0], p[1] / 2, p[2]]}>
        <cylinderGeometry args={[0.04, 0.04, Math.max(p[1], 0.1), 6]} />
        <meshBasicMaterial color="#555" />
      </mesh>
      <mesh position={p}>
        <sphereGeometry args={[0.45, 16, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={lt.coasting ? 0.1 : 0.8} transparent opacity={alpha} />
      </mesh>
      {associated && (
        <mesh position={p}>
          <torusGeometry args={[0.9, 0.06, 8, 32]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
      )}
      <Html position={[p[0], p[1] + 0.8, p[2]]} center distanceFactor={25} style={{ pointerEvents: "none" }}>
        <div className="label3d" style={{ color }}>{`#${lt.track_id} ${lt.distance_m}m`}</div>
      </Html>
    </group>
  );
}

function EgoCar() {
  return (
    <group position={[0, 0.7, 1.0]}>
      <mesh>
        <boxGeometry args={[1.8, 1.4, 4.4]} />
        <meshStandardMaterial color="#4da3ff" transparent opacity={0.6} />
      </mesh>
      <mesh position={[0, 0.2, -2.3]}>
        <coneGeometry args={[0.4, 0.8, 4]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>
    </group>
  );
}

export default function Scene3D({ state, cloud }: { state: WorldState | null; cloud: Cloud | null }) {
  const camPos = useMemo<[number, number, number]>(() => [8, 12, 18], []);
  const assoc = state?.association.ego_lane_light_track_id ?? null;
  const lostRef = useRef(0);
  return (
    <div className="panel scene-panel">
      <Canvas
        camera={{ position: camPos, fov: 55, near: 0.1, far: 500 }}
        dpr={1}
        gl={{ powerPreference: "low-power", antialias: false, preserveDrawingBuffer: false }}
        onCreated={({ gl }) => {
          gl.domElement.addEventListener("webglcontextlost", (e) => {
            e.preventDefault();
            lostRef.current++;
          });
        }}
      >
        <color attach="background" args={["#0e1116"]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 20, 10]} intensity={0.8} />
        <Grid args={[120, 120]} cellSize={1} sectionSize={10} cellColor="#233" sectionColor="#3a5a6a" fadeDistance={90} infiniteGrid position={[0, -0.01, 0]} />
        <EgoCar />
        {state?.lanes.map((lane) => <LaneItem key={lane.id} lane={lane} />)}
        {state?.traffic_lights.map((lt) => <LightItem key={lt.track_id} lt={lt} associated={lt.track_id === assoc} />)}
        <PointCloud cloud={cloud} />
        <OrbitControls target={[0, 0, -15]} maxPolarAngle={Math.PI / 2 - 0.02} enableDamping />
      </Canvas>
      {!state && <div className="placeholder">等待状态…</div>}
    </div>
  );
}
