import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Grid, Line, OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import type { Cloud, WorldState } from "../types";

/** ego (x fwd, y left, z up) -> three (x right, y up, z toward viewer) */
const toThree = (p: [number, number, number] | number[]): [number, number, number] => [-p[1], p[2], -p[0]];

const LAMP_COLORS: Record<string, string> = { red: "#ff3b30", yellow: "#ffcc00", green: "#34c759", empty: "#666" };

function PointCloud({ cloud }: { cloud: Cloud | null }) {
  const geomRef = useRef<THREE.BufferGeometry>(null);
  useEffect(() => {
    const g = geomRef.current;
    if (!g || !cloud) return;
    const n = cloud.n;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const x = cloud.xyz[i * 3], y = cloud.xyz[i * 3 + 1], z = cloud.xyz[i * 3 + 2];
      pos[i * 3] = -y;
      pos[i * 3 + 1] = z;
      pos[i * 3 + 2] = -x;
      col[i * 3] = cloud.rgb[i * 3] / 255;
      col[i * 3 + 1] = cloud.rgb[i * 3 + 1] / 255;
      col[i * 3 + 2] = cloud.rgb[i * 3 + 2] / 255;
    }
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.computeBoundingSphere();
  }, [cloud]);
  return (
    <points frustumCulled={false}>
      <bufferGeometry ref={geomRef} />
      <pointsMaterial size={0.12} vertexColors sizeAttenuation />
    </points>
  );
}

function Lanes({ state }: { state: WorldState }) {
  return (
    <>
      {state.lanes.map((lane) => {
        const pts = lane.points_3d.map(toThree);
        if (pts.length < 2) return null;
        const color = lane.coasting ? "#9a9a9a" : lane.role.startsWith("ego") ? "#34c759" : "#ffb340";
        return <Line key={lane.id} points={pts} color={color} lineWidth={lane.role.startsWith("ego") ? 4 : 2} transparent opacity={0.4 + 0.6 * lane.confidence} />;
      })}
    </>
  );
}

function Lights({ state }: { state: WorldState }) {
  const assoc = state.association.ego_lane_light_track_id;
  return (
    <>
      {state.traffic_lights.map((lt) => {
        const p = toThree(lt.position_ego);
        const main = lt.lamps.find((l) => l.shape === "circle") ?? lt.lamps[0];
        const color = lt.coasting ? "#777" : main ? LAMP_COLORS[main.color] ?? "#fff" : "#fff";
        const alpha = 0.3 + 0.7 * Math.min(1, lt.distance_confidence);
        return (
          <group key={lt.track_id}>
            <Line points={[[p[0], 0, p[2]], p]} color="#555" lineWidth={1} />
            <mesh position={p}>
              <sphereGeometry args={[0.45, 16, 16]} />
              <meshStandardMaterial color={color} emissive={color} emissiveIntensity={lt.coasting ? 0.1 : 0.8} transparent opacity={alpha} />
            </mesh>
            {lt.track_id === assoc && (
              <mesh position={p}>
                <torusGeometry args={[0.9, 0.06, 8, 32]} />
                <meshBasicMaterial color="#ffffff" />
              </mesh>
            )}
            <Text position={[p[0], p[1] + 0.9, p[2]]} fontSize={0.6} color="#fff" anchorX="center" anchorY="bottom">
              {`#${lt.track_id} ${lt.distance_m}m`}
            </Text>
          </group>
        );
      })}
    </>
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

function AutoFollow({ enabled }: { enabled: boolean }) {
  // keeps the camera target on the ego car; OrbitControls handles rotation/zoom
  useFrame(() => void enabled);
  return null;
}

export default function Scene3D({ state, cloud }: { state: WorldState | null; cloud: Cloud | null }) {
  const camPos = useMemo<[number, number, number]>(() => [8, 12, 18], []);
  return (
    <div className="panel scene-panel">
      <Canvas camera={{ position: camPos, fov: 55, near: 0.1, far: 500 }} dpr={[1, 1.5]}>
        <color attach="background" args={["#0e1116"]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 20, 10]} intensity={0.8} />
        <Grid args={[120, 120]} cellSize={1} sectionSize={10} cellColor="#233" sectionColor="#3a5a6a" fadeDistance={90} infiniteGrid position={[0, -0.01, 0]} />
        <EgoCar />
        {state && <Lanes state={state} />}
        {state && <Lights state={state} />}
        <PointCloud cloud={cloud} />
        <OrbitControls target={[0, 0, -15]} maxPolarAngle={Math.PI / 2 - 0.02} enableDamping />
        <AutoFollow enabled />
      </Canvas>
      {!state && <div className="placeholder">等待状态…</div>}
    </div>
  );
}
