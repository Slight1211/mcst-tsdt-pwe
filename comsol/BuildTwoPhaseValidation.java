import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/** Two-phase epoxy/steel validation. Inner and outer inclusion partitions
 * have identical steel properties; their interface is not a material boundary.
 * MCST is applied only to the epoxy matrix. No piezo or air in this batch. */
public class BuildTwoPhaseValidation {
  private static int envInt(String name, int fallback) {
    String value = System.getenv(name);
    return value == null || value.isEmpty() ? fallback : Integer.parseInt(value);
  }

  private static double envDouble(String name, double fallback) {
    String value = System.getenv(name);
    return value == null || value.isEmpty() ? fallback : Double.parseDouble(value);
  }

  private static String envString(String name, String fallback) {
    String value = System.getenv(name);
    return value == null || value.isEmpty() ? fallback : value;
  }

  private static double[] envDoubleArray(String name, double[] fallback) {
    String value = System.getenv(name);
    if (value == null || value.trim().isEmpty()) {
      return fallback;
    }
    String[] fields = value.split(",");
    if (fields.length != fallback.length) {
      throw new IllegalArgumentException(
          name + " must contain " + fallback.length + " comma-separated values");
    }
    double[] result = new double[fields.length];
    for (int index = 0; index < fields.length; index++) {
      result[index] = Double.parseDouble(fields[index].trim());
    }
    return result;
  }

  public static Model run() {
    long classStart = System.nanoTime();
    Model model = ModelUtil.create("Model");
    model.modelPath(envString("COMSOL_MODEL_DIR", "."));
    model.label("concentric_pzt_bloch.mph");
    model.title("Constant-thickness concentric PZT local-resonance Bloch cell");

    int meshLevel = envInt("COMSOL_MESH_LEVEL", 5);
    int pztLayers = envInt("COMSOL_PZT_SWEEP_ELEMENTS", 2);
    int pathPoints = envInt("COMSOL_PATH_POINTS_PER_SEGMENT", 0);
    int ibzSubdivisions = envInt("COMSOL_IBZ_SUBDIVISIONS", 0);
    int eigenCount = envInt("COMSOL_EIGEN_COUNT", 12);
    double eigenShiftMhz = envDouble("COMSOL_EIGEN_SHIFT_MHZ", 2.0);
    double latticeUm = envDouble("COMSOL_LATTICE_UM", 100.0);
    double coreUm = envDouble("COMSOL_CORE_RADIUS_UM", 20.0);
    double ringUm = envDouble("COMSOL_RING_RADIUS_UM", 34.0);
    double splitUm = envDouble("COMSOL_ELECTRODE_SPLIT_UM", 27.0);
    double thicknessUm = envDouble("COMSOL_THICKNESS_UM", 8.0);
    double pztUm = envDouble("COMSOL_PZT_FACE_UM", 0.8);
    double epoxyGpa = envDouble("COMSOL_RING_E_GPA", 0.5);
    double mcstUm = envDouble("COMSOL_MCST_LENGTH_UM", 0.0);
    boolean nodeGeometry = envString(
        "COMSOL_NODE_GEOMETRY", "false").equalsIgnoreCase("true");
    String electricalCase = envString("COMSOL_ELECTRICAL_CASE", "short").toLowerCase();
    String kx = envString("COMSOL_KX", "pi/a");
    String ky = envString("COMSOL_KY", "0[1/m]");
    String resultCsv = envString(
        "COMSOL_RESULT_CSV",
        "concentric_comsol_single.csv");
    String timingCsv = envString(
        "COMSOL_TIMING_CSV",
        "concentric_comsol_single_timing.csv");
    boolean mcstEnabled = mcstUm > 0.0;
    boolean auditPair = envString(
        "COMSOL_MCST_AUDIT_PAIR", "false").equalsIgnoreCase("true");

    if (!electricalCase.equals("short") && !electricalCase.equals("open")) {
      throw new IllegalArgumentException("Electrical case must be short or open");
    }
    if (!(0.0 < coreUm && coreUm < splitUm && splitUm < ringUm
        && ringUm < 0.5 * latticeUm)) {
      throw new IllegalArgumentException("Require 0 < core < split < ring < a/2");
    }
    if (!(pztUm >= 0.0 && 2.0 * pztUm < thicknessUm)) {
      throw new IllegalArgumentException("Require 0 <= 2*tp < h");
    }
    if (nodeGeometry && pztUm > 0.0) {
      throw new IllegalArgumentException(
          "The optimized mother-cell audit uses through-thickness materials and no PZT face layers");
    }
    if (pathPoints > 0 && ibzSubdivisions > 0) {
      throw new IllegalArgumentException("Path and IBZ sweeps are mutually exclusive");
    }
    if (auditPair && (!mcstEnabled || pathPoints > 0 || ibzSubdivisions > 0)) {
      throw new IllegalArgumentException("MCST pair requires l>0 and one Bloch point");
    }

    model.param().set("a", Double.toString(latticeUm) + "[um]", "Lattice constant");
    model.param().set("hp", Double.toString(thicknessUm) + "[um]", "Total thickness");
    model.param().set("ri", Double.toString(coreUm) + "[um]", "Tungsten core radius");
    model.param().set("ro", Double.toString(ringUm) + "[um]", "Compliant-ring outer radius");
    model.param().set("rs", Double.toString(splitUm) + "[um]", "Electrode split radius");
    model.param().set("tp", Double.toString(pztUm) + "[um]", "Each PZT face thickness");
    model.param().set("Ee", Double.toString(epoxyGpa) + "[GPa]", "Compliant-ring modulus");
    model.param().set("kx", kx, "Bloch wave number x");
    model.param().set("ky", ky, "Bloch wave number y");
    model.param().set("Esi", "4.35[GPa]", "Silicon Young modulus");
    model.param().set("nusi", "4.35/(2*1.59)-1", "Silicon Poisson ratio");
    model.param().set("muSi", "Esi/(2*(1+nusi))", "Silicon shear modulus");
    if (mcstEnabled) {
      model.param().set("ellMcst", Double.toString(mcstUm) + "[um]", "MCST length");
    }

    model.component().create("comp1", true);
    model.component("comp1").sorder("quadratic");
    model.component("comp1").geom().create("geom1", 3);
    model.component("comp1").geom("geom1").lengthUnit("um");

    model.component("comp1").geom("geom1").create("host", "Block");
    model.component("comp1").geom("geom1").feature("host").set("base", "center");
    model.component("comp1").geom("geom1").feature("host")
        .set("size", new String[]{"a", "a", "hp"});

    double[] optimizedCoreRadii = envDoubleArray(
        "COMSOL_NODE_CORE_RADII", new double[]{
          0.2507812378448442, 0.24595179470501452,
          0.2560766721624591, 0.2538164812546278});
    double[] optimizedRingRadii = envDoubleArray(
        "COMSOL_NODE_RING_RADII", new double[]{
          0.4630809080006034, 0.4399392039326427,
          0.4123795780020756, 0.36409279546416906});
    double[][] coreNodes = c4vNodeTable(optimizedCoreRadii, latticeUm);
    double[][] ringNodes = c4vNodeTable(optimizedRingRadii, latticeUm);

    if (nodeGeometry) {
      createPolygonPrism(
          model, "ringCut", ringNodes, "hp+2[um]", "-hp/2-1[um]", false);
    } else {
      createCylinder(model, "ringCut", "ro", "hp+2[um]", "-hp/2-1[um]", false);
    }
    model.component("comp1").geom("geom1").create("matrix", "Difference");
    model.component("comp1").geom("geom1").feature("matrix")
        .selection("input").set("host");
    model.component("comp1").geom("geom1").feature("matrix")
        .selection("input2").set("ringCut");
    model.component("comp1").geom("geom1").feature("matrix").set("selresult", true);

    if (pztUm > 0.0) {
      createAnnulus(model, "pztBottom", "ro", "ri", "tp", "-hp/2", true);
      createAnnulus(model, "epoxyRing", "ro", "ri", "hp-2*tp", "-hp/2+tp", true);
      createAnnulus(model, "pztTop", "ro", "ri", "tp", "hp/2-tp", true);
    } else if (nodeGeometry) {
      createPolygonAnnulus(
          model, "epoxyRing", ringNodes, coreNodes, "hp", "-hp/2", true);
    } else {
      createAnnulus(model, "epoxyRing", "ro", "ri", "hp", "-hp/2", true);
    }
    if (nodeGeometry) {
      createPolygonPrism(model, "core", coreNodes, "hp", "-hp/2", true);
    } else {
      createCylinder(model, "core", "ri", "hp", "-hp/2", true);
    }

    if (pztUm > 0.0) {
      model.component("comp1").geom("geom1").create("pztStack", "Union");
      model.component("comp1").geom("geom1").feature("pztStack")
          .selection("input").set(new String[]{"pztBottom", "pztTop"});
      model.component("comp1").geom("geom1").feature("pztStack").set("intbnd", true);
      model.component("comp1").geom("geom1").feature("pztStack").set("selresult", true);
    }

    model.component("comp1").geom("geom1").create("uni1", "Union");
    if (pztUm > 0.0) {
      model.component("comp1").geom("geom1").feature("uni1")
          .selection("input").set(
              new String[]{"matrix", "pztStack", "epoxyRing", "core"});
    } else {
      model.component("comp1").geom("geom1").feature("uni1")
          .selection("input").set(new String[]{"matrix", "epoxyRing", "core"});
    }
    model.component("comp1").geom("geom1").feature("uni1").set("intbnd", true);
    model.component("comp1").geom("geom1").run();

    double halfA = 0.5 * latticeUm;
    createFaceBox(model, "selXL", -halfA - 0.001, -halfA + 0.001,
        -halfA - 0.001, halfA + 0.001,
        -0.5 * thicknessUm - 0.001, 0.5 * thicknessUm + 0.001);
    createFaceBox(model, "selXR", halfA - 0.001, halfA + 0.001,
        -halfA - 0.001, halfA + 0.001,
        -0.5 * thicknessUm - 0.001, 0.5 * thicknessUm + 0.001);
    createFaceBox(model, "selYF", -halfA - 0.001, halfA + 0.001,
        -halfA - 0.001, -halfA + 0.001,
        -0.5 * thicknessUm - 0.001, 0.5 * thicknessUm + 0.001);
    createFaceBox(model, "selYB", -halfA - 0.001, halfA + 0.001,
        halfA - 0.001, halfA + 0.001,
        -0.5 * thicknessUm - 0.001, 0.5 * thicknessUm + 0.001);
    if (pztUm > 0.0) {
      createFaceBox(model, "selBottomOuter", -ringUm - 0.001, ringUm + 0.001,
          -ringUm - 0.001, ringUm + 0.001,
          -0.5 * thicknessUm - 0.001, -0.5 * thicknessUm + 0.001);
      createFaceBox(model, "selBottomInner", -ringUm - 0.001, ringUm + 0.001,
          -ringUm - 0.001, ringUm + 0.001,
          -0.5 * thicknessUm + pztUm - 0.001,
          -0.5 * thicknessUm + pztUm + 0.001);
      createFaceBox(model, "selTopInner", -ringUm - 0.001, ringUm + 0.001,
          -ringUm - 0.001, ringUm + 0.001,
          0.5 * thicknessUm - pztUm - 0.001,
          0.5 * thicknessUm - pztUm + 0.001);
      createFaceBox(model, "selTopOuter", -ringUm - 0.001, ringUm + 0.001,
          -ringUm - 0.001, ringUm + 0.001,
          0.5 * thicknessUm - 0.001, 0.5 * thicknessUm + 0.001);
    }
    createUnion(model, "selXPair", new String[]{"selXL", "selXR"}, 2);
    createUnion(model, "selYPair", new String[]{"selYF", "selYB"}, 2);
    if (pztUm > 0.0) {
      createUnion(model, "selPztGround", new String[]{"selBottomInner", "selTopInner"}, 2);
      createUnion(model, "selPztTerminal", new String[]{"selBottomOuter", "selTopOuter"}, 2);
    }

    model.component("comp1").cpl().create("intSolid", "Integration");
    model.component("comp1").cpl("intSolid").selection().all();
    model.component("comp1").cpl().create("intCore", "Integration");
    model.component("comp1").cpl("intCore").selection().named("geom1_core_dom");
    if (mcstEnabled) {
      model.component("comp1").cpl().create("intHost", "Integration");
      model.component("comp1").cpl("intHost").selection().named("geom1_matrix_dom");
    }

    addIsotropicMaterial(model, "matSi", "Epoxy matrix", "geom1_matrix_dom",
        "4.35[GPa]", "4.35/(2*1.59)-1", "1180[kg/m^3]");
    addIsotropicMaterial(model, "matEpoxy", "Steel outer partition",
        "geom1_epoxyRing_dom", "210.6[GPa]", "0.3", "7780[kg/m^3]");
    addIsotropicMaterial(model, "matW", "Steel inner partition", "geom1_core_dom",
        "210.6[GPa]", "0.3", "7780[kg/m^3]");
    // Both domains use the same global material orientation.  Because the
    // outer electrodes share one voltage and the inner electrodes are
    // grounded, E_z has opposite sign in the two layers and bending charges add.
    if (pztUm > 0.0) {
      addPzt5hMaterial(model, "matPztBottom", "geom1_pztBottom_dom", 1.0);
      addPzt5hMaterial(model, "matPztTop", "geom1_pztTop_dom", 1.0);
    }

    model.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
    if (pztUm > 0.0) {
      model.component("comp1").physics("solid")
          .create("pzm1", "PiezoelectricMaterialModel", 3);
      model.component("comp1").physics("solid").feature("pzm1")
          .selection().named("geom1_pztStack_dom");
    }
    addFloquetPair(model, "solid", "pcx", "selXPair");
    addFloquetPair(model, "solid", "pcy", "selYPair");
    if (mcstEnabled) addMcstSiliconWeakForm(model);

    if (pztUm > 0.0) {
      model.component("comp1").physics().create("es", "Electrostatics", "geom1");
      model.component("comp1").physics("es").selection().named("geom1_pztStack_dom");
      model.component("comp1").physics("es").create("ccnp1", "ChargeConservationPiezo", 3);
      model.component("comp1").physics("es").feature("ccnp1").selection().all();
      model.component("comp1").physics("es").create("gnd1", "Ground", 2);
      model.component("comp1").physics("es").feature("gnd1")
          .selection().named("selPztGround");
      if (electricalCase.equals("short")) {
        model.component("comp1").physics("es").create("gnd2", "Ground", 2);
        model.component("comp1").physics("es").feature("gnd2")
            .selection().named("selPztTerminal");
      } else {
        model.component("comp1").physics("es").create("term1", "Terminal", 2);
        model.component("comp1").physics("es").feature("term1")
            .selection().named("selPztTerminal");
        model.component("comp1").physics("es").feature("term1")
            .set("TerminalType", "Charge");
        model.component("comp1").physics("es").feature("term1").set("Q0", "0[C]");
      }
      model.component("comp1").multiphysics().create("pze1", "PiezoelectricEffect", 3);
      model.component("comp1").multiphysics("pze1").set("InitializePiezoCoupling", 1);
    }

    model.component("comp1").mesh().create("mesh1");
    model.component("comp1").mesh("mesh1").create("size1", "Size");
    model.component("comp1").mesh("mesh1").feature("size1").set("hauto", meshLevel);
    int drySweepLayers = envInt("COMSOL_DRY_SWEEP_LAYERS", 0);
    if (drySweepLayers > 0) {
      if (pztUm > 0.0) throw new IllegalArgumentException("Dry sweep requires no piezo layers");
      createFaceBox(model, "selDryBottom", -latticeUm, latticeUm, -latticeUm, latticeUm,
          -thicknessUm/2-0.0001, -thicknessUm/2+0.0001);
      createFaceBox(model, "selDryTop", -latticeUm, latticeUm, -latticeUm, latticeUm,
          thicknessUm/2-0.0001, thicknessUm/2+0.0001);
      model.component("comp1").mesh("mesh1").create("dryTri", "FreeTri");
      model.component("comp1").mesh("mesh1").feature("dryTri").selection().named("selDryBottom");
      model.component("comp1").mesh("mesh1").create("drySweep", "Sweep");
      model.component("comp1").mesh("mesh1").feature("drySweep").selection().geom("geom1",3);
      model.component("comp1").mesh("mesh1").feature("drySweep").selection().all();
      model.component("comp1").mesh("mesh1").feature("drySweep").selection("sourceface").named("selDryBottom");
      model.component("comp1").mesh("mesh1").feature("drySweep").selection("targetface").named("selDryTop");
      model.component("comp1").mesh("mesh1").feature("drySweep").create("dis1", "Distribution");
      model.component("comp1").mesh("mesh1").feature("drySweep").feature("dis1").set("numelem",drySweepLayers);
    } else {
    model.component("comp1").mesh("mesh1").create("ftriX", "FreeTri");
    model.component("comp1").mesh("mesh1").feature("ftriX").selection().named("selXL");
    model.component("comp1").mesh("mesh1").create("copyX", "CopyFace");
    model.component("comp1").mesh("mesh1").feature("copyX")
        .selection("source").named("selXL");
    model.component("comp1").mesh("mesh1").feature("copyX")
        .selection("destination").named("selXR");
    model.component("comp1").mesh("mesh1").create("ftriY", "FreeTri");
    model.component("comp1").mesh("mesh1").feature("ftriY").selection().named("selYF");
    model.component("comp1").mesh("mesh1").create("copyY", "CopyFace");
    model.component("comp1").mesh("mesh1").feature("copyY")
        .selection("source").named("selYF");
    model.component("comp1").mesh("mesh1").feature("copyY")
        .selection("destination").named("selYB");
    if (pztUm > 0.0 && pztLayers > 0) {
      model.component("comp1").mesh("mesh1").create("sweBottom", "Sweep");
      model.component("comp1").mesh("mesh1").feature("sweBottom")
          .selection().named("geom1_pztBottom_dom");
      model.component("comp1").mesh("mesh1").feature("sweBottom")
          .selection("sourceface").named("selBottomOuter");
      model.component("comp1").mesh("mesh1").feature("sweBottom")
          .selection("targetface").named("selBottomInner");
      model.component("comp1").mesh("mesh1").feature("sweBottom")
          .create("dis1", "Distribution");
      model.component("comp1").mesh("mesh1").feature("sweBottom")
          .feature("dis1").set("numelem", pztLayers);
      model.component("comp1").mesh("mesh1").create("sweTop", "Sweep");
      model.component("comp1").mesh("mesh1").feature("sweTop")
          .selection().named("geom1_pztTop_dom");
      model.component("comp1").mesh("mesh1").feature("sweTop")
          .selection("sourceface").named("selTopInner");
      model.component("comp1").mesh("mesh1").feature("sweTop")
          .selection("targetface").named("selTopOuter");
      model.component("comp1").mesh("mesh1").feature("sweTop")
          .create("dis1", "Distribution");
      model.component("comp1").mesh("mesh1").feature("sweTop")
          .feature("dis1").set("numelem", pztLayers);
    }
    model.component("comp1").mesh("mesh1").create("ftet1", "FreeTet");
    }
    model.component("comp1").mesh("mesh1").run();

    model.study().create("std1");
    model.study("std1").create("eig", "Eigenfrequency");
    model.study("std1").feature("eig").set("neigs", eigenCount);
    model.study("std1").feature("eig").set(
        "shift", Double.toString(eigenShiftMhz) + "[MHz]");

    double solveSeconds;
    int solvedPoints;
    if (auditPair) {
      solveSeconds = runClassicalMcstPair(model, resultCsv);
      solvedPoints = 1;
    } else if (pathPoints > 0) {
      solveSeconds = runBlochPath(model, pathPoints, resultCsv, mcstEnabled);
      solvedPoints = 3 * pathPoints + 1;
    } else if (ibzSubdivisions > 0) {
      solveSeconds = runBlochIbz(model, ibzSubdivisions, resultCsv, mcstEnabled);
      solvedPoints = (ibzSubdivisions + 1) * (ibzSubdivisions + 2) / 2;
      if (envInt("COMSOL_CONTROL_XM_ONLY", 0) == 1) solvedPoints = 2;
    } else {
      long start = System.nanoTime();
      model.study("std1").run();
      solveSeconds = (System.nanoTime() - start) / 1.0e9;
      solvedPoints = 1;
      createFrequencyEvaluation(model, mcstEnabled);
      model.result().table().create("tblFreq", "Table");
      model.result().numerical("gevFreq").set("table", "tblFreq");
      model.result().numerical("gevFreq").setResult();
      model.result().table("tblFreq").save(resultCsv);
    }

    double classSeconds = (System.nanoTime() - classStart) / 1.0e9;
    PrintWriter timingWriter = null;
    try {
      timingWriter = new PrintWriter(timingCsv);
      timingWriter.println("mesh_level,pzt_layers,electrical_case,mcst_length_um,a_um,core_um,split_um,ring_um,h_um,tp_um,epoxy_gpa,kx,ky,points,eigen_count,eigen_shift_mhz,solve_seconds,class_seconds");
      timingWriter.println(meshLevel + "," + pztLayers + "," + electricalCase + ","
          + mcstUm + "," + latticeUm + "," + coreUm + "," + splitUm + "," + ringUm + ","
          + thicknessUm + "," + pztUm + "," + epoxyGpa + ",\"" + kx
          + "\",\"" + ky + "\"," + solvedPoints + "," + eigenCount + ","
          + eigenShiftMhz + "," + solveSeconds + "," + classSeconds);
      timingWriter.close();
    } catch (Exception exception) {
      if (timingWriter != null) timingWriter.close();
      throw new RuntimeException("Could not save timing metadata", exception);
    }
    return model;
  }

  private static void createFrequencyEvaluation(Model model, boolean includeMcst) {
    try {
      model.result().numerical().remove("gevFreq");
    } catch (Exception ignored) {}
    model.result().numerical().create("gevFreq", "EvalGlobal");
    model.result().numerical("gevFreq").set("data", "dset1");
    String kinetic = "intSolid(solid.rho*(abs(u)^2+abs(v)^2+abs(w)^2))";
    List<String> expressions = new ArrayList<String>();
    List<String> units = new ArrayList<String>();
    List<String> descriptions = new ArrayList<String>();
    expressions.add("freq");
    expressions.add("intSolid(solid.rho*abs(w)^2)/(" + kinetic + ")");
    expressions.add("intCore(solid.rho*(abs(u)^2+abs(v)^2+abs(w)^2))/(" + kinetic + ")");
    expressions.add("intCore(solid.rho*abs(w)^2)/(" + kinetic + ")");
    units.add("MHz"); units.add("1"); units.add("1"); units.add("1");
    descriptions.add("Eigenfrequency");
    descriptions.add("Vertical kinetic-energy fraction");
    descriptions.add("Tungsten-core kinetic-energy fraction");
    descriptions.add("Tungsten-core vertical kinetic-energy fraction");
    if (includeMcst) {
      expressions.add("sqrt(intHost(mcst_constraint_sq)/intHost(1))");
      expressions.add("2*intHost(mcst_curv_energy)/((2*pi*freq)^2*(" + kinetic + "))");
      units.add("1"); units.add("1");
      descriptions.add("MCST rotation-constraint RMS");
      descriptions.add("MCST curvature-energy fraction");
    }
    model.result().numerical("gevFreq").set(
        "expr", expressions.toArray(new String[expressions.size()]));
    model.result().numerical("gevFreq").set(
        "unit", units.toArray(new String[units.size()]));
    model.result().numerical("gevFreq").set(
        "descr", descriptions.toArray(new String[descriptions.size()]));
  }

  private static double runClassicalMcstPair(Model model, String resultCsv) {
    model.study("std1").feature("eig").activate("mcstRot", false);
    model.study("std1").feature("eig").activate("mcstLag", false);
    long start = System.nanoTime();
    model.study("std1").run();
    double classicalSeconds = (System.nanoTime() - start) / 1.0e9;
    createFrequencyEvaluation(model, false);
    double[][] cr = model.result().numerical("gevFreq").getReal();
    double[][] ci = model.result().numerical("gevFreq").getImag();
    model.study("std1").feature("eig").activate("mcstRot", true);
    model.study("std1").feature("eig").activate("mcstLag", true);
    start = System.nanoTime();
    model.study("std1").run();
    double mcstSeconds = (System.nanoTime() - start) / 1.0e9;
    createFrequencyEvaluation(model, true);
    double[][] mr = model.result().numerical("gevFreq").getReal();
    double[][] mi = model.result().numerical("gevFreq").getImag();
    int count = Math.min(cr[0].length, mr[0].length);
    PrintWriter writer = null;
    try {
      writer = new PrintWriter(resultCsv);
      writer.println("mode_index,classical_frequency_mhz,classical_imag_mhz,mcst_frequency_mhz,mcst_imag_mhz,relative_shift_percent,classical_vertical_fraction,mcst_vertical_fraction,mcst_constraint_rms,mcst_curvature_fraction");
      for (int mode = 0; mode < count; ++mode) {
        double shift = 100.0 * (mr[0][mode] - cr[0][mode]) / cr[0][mode];
        writer.println((mode + 1) + "," + cr[0][mode] + "," + ci[0][mode]
            + "," + mr[0][mode] + "," + mi[0][mode] + "," + shift + ","
            + cr[1][mode] + "," + mr[1][mode] + "," + mr[4][mode] + ","
            + mr[5][mode]);
      }
      writer.close();
    } catch (Exception exception) {
      if (writer != null) writer.close();
      throw new RuntimeException("Could not save MCST pair", exception);
    }
    return classicalSeconds + mcstSeconds;
  }

  private static double runBlochPath(
      Model model, int pointsPerSegment, String resultCsv, boolean includeMcst) {
    double[][] vertices = {{0, 0}, {1, 0}, {1, 1}, {0, 0}};
    int total = 3 * pointsPerSegment + 1;
    double totalSeconds = 0.0;
    PrintWriter writer = null;
    try {
      writer = new PrintWriter(resultCsv);
      writer.println("point_index,kx_pi_over_a,ky_pi_over_a,mode_index,frequency_mhz,imag_frequency_mhz,vertical_fraction,core_fraction,core_vertical_fraction,solve_seconds");
      int point = 0;
      for (int segment = 0; segment < 3; ++segment) {
        for (int local = 0; local < pointsPerSegment; ++local) {
          double fraction = ((double) local) / pointsPerSegment;
          double px = vertices[segment][0] + fraction * (vertices[segment + 1][0] - vertices[segment][0]);
          double py = vertices[segment][1] + fraction * (vertices[segment + 1][1] - vertices[segment][1]);
          totalSeconds += solveAndWritePoint(model, writer, point, px, py, includeMcst);
          ++point;
        }
      }
      totalSeconds += solveAndWritePoint(model, writer, point, 0.0, 0.0, includeMcst);
      writer.close();
    } catch (Exception exception) {
      if (writer != null) writer.close();
      throw new RuntimeException("Could not save Bloch path", exception);
    }
    return totalSeconds;
  }

  private static double runBlochIbz(
      Model model, int subdivisions, String resultCsv, boolean includeMcst) {
    double totalSeconds = 0.0;
    PrintWriter writer = null;
    try {
      writer = new PrintWriter(resultCsv);
      writer.println("point_index,kx_pi_over_a,ky_pi_over_a,mode_index,frequency_mhz,imag_frequency_mhz,vertical_fraction,core_fraction,core_vertical_fraction,solve_seconds");
      int point = 0;
      for (int ix = 0; ix <= subdivisions; ++ix) {
        if (envInt("COMSOL_CONTROL_XM_ONLY", 0) == 1) {
          if (subdivisions != 1) throw new IllegalArgumentException("XM-only requires IBZ subdivisions=1");
          if (ix == 0) continue;
        }
        for (int iy = 0; iy <= ix; ++iy) {
          totalSeconds += solveAndWritePoint(
              model, writer, point, ((double) ix) / subdivisions,
              ((double) iy) / subdivisions, includeMcst);
          ++point;
        }
      }
      writer.close();
    } catch (Exception exception) {
      if (writer != null) writer.close();
      throw new RuntimeException("Could not save IBZ grid", exception);
    }
    return totalSeconds;
  }

  private static double solveAndWritePoint(
      Model model, PrintWriter writer, int point, double px, double py,
      boolean includeMcst) {
    model.param().set("kx", Double.toString(px) + "*pi/a");
    model.param().set("ky", Double.toString(py) + "*pi/a");
    long start = System.nanoTime();
    model.study("std1").run();
    double seconds = (System.nanoTime() - start) / 1.0e9;
    createFrequencyEvaluation(model, includeMcst);
    double[][] real = model.result().numerical("gevFreq").getReal();
    double[][] imag = model.result().numerical("gevFreq").getImag();
    for (int mode = 0; mode < real[0].length; ++mode) {
      writer.println(point + "," + px + "," + py + "," + (mode + 1) + ","
          + real[0][mode] + "," + imag[0][mode] + "," + real[1][mode] + ","
          + real[2][mode] + "," + real[3][mode] + "," + seconds);
    }
    writer.flush();
    System.out.println("Bloch point " + point + " solved in " + seconds + " s");
    return seconds;
  }

  private static void addMcstSiliconWeakForm(Model model) {
    String selection = "geom1_matrix_dom";
    model.component("comp1").physics().create("mcstRot", "SolidMechanics", "geom1");
    model.component("comp1").physics("mcstRot").field("displacement")
        .component(new String[]{"mrx", "mry", "mrz"});
    model.component("comp1").physics("mcstRot").selection().named(selection);
    zeroAuxiliarySolidConstitutive(model, "mcstRot");
    model.component("comp1").physics("mcstRot").prop("ShapeProperty")
        .set("order_displacement", 2);
    model.component("comp1").physics().create("mcstLag", "SolidMechanics", "geom1");
    model.component("comp1").physics("mcstLag").field("displacement")
        .component(new String[]{"mlx", "mly", "mlz"});
    model.component("comp1").physics("mcstLag").selection().named(selection);
    zeroAuxiliarySolidConstitutive(model, "mcstLag");
    model.component("comp1").physics("mcstLag").prop("ShapeProperty")
        .set("order_displacement", 1);
    addFloquetPair(model, "mcstRot", "pcx", "selXPair");
    addFloquetPair(model, "mcstRot", "pcy", "selYPair");
    addFloquetPair(model, "mcstLag", "pcx", "selXPair");
    addFloquetPair(model, "mcstLag", "pcy", "selYPair");
    String chi2 = "d(mrx,x)^2+d(mry,y)^2+d(mrz,z)^2"
        + "+2*(0.5*(d(mrx,y)+d(mry,x)))^2"
        + "+2*(0.5*(d(mrx,z)+d(mrz,x)))^2"
        + "+2*(0.5*(d(mry,z)+d(mrz,y)))^2";
    String pr = "mlx*mrx+mly*mry+mlz*mrz";
    String pcurl = "mlx*(d(w,y)-d(v,z))+mly*(d(u,z)-d(w,x))"
        + "+mlz*(d(v,x)-d(u,y))";
    String energy = "muSi*(" + chi2 + ")+muSi/ellMcst^2*(" + pr + ")"
        + "-muSi/(2*ellMcst)*(" + pcurl + ")";
    model.component("comp1").physics("mcstRot")
        .create("weakMcst", "WeakContribution", 3);
    model.component("comp1").physics("mcstRot").feature("weakMcst")
        .selection().named(selection);
    model.component("comp1").physics("mcstRot").feature("weakMcst")
        .set("weakExpression", "-test(" + energy + ")");
    model.component("comp1").physics("mcstRot").feature("weakMcst")
        .set("integrationOrder", 4);
    model.component("comp1").variable().create("varMcst");
    model.component("comp1").variable("varMcst").selection().named(selection);
    model.component("comp1").variable("varMcst").set("mcst_constraint_sq",
        "abs(mrx/ellMcst-0.5*(d(w,y)-d(v,z)))^2"
        + "+abs(mry/ellMcst-0.5*(d(u,z)-d(w,x)))^2"
        + "+abs(mrz/ellMcst-0.5*(d(v,x)-d(u,y)))^2");
    model.component("comp1").variable("varMcst").set("mcst_curv_energy",
        "muSi*(abs(d(mrx,x))^2+abs(d(mry,y))^2+abs(d(mrz,z))^2"
        + "+2*abs(0.5*(d(mrx,y)+d(mry,x)))^2"
        + "+2*abs(0.5*(d(mrx,z)+d(mrz,x)))^2"
        + "+2*abs(0.5*(d(mry,z)+d(mrz,y)))^2)");
  }

  private static void addFloquetPair(
      Model model, String physics, String tag, String selection) {
    model.component("comp1").physics(physics).create(tag, "PeriodicCondition", 2);
    model.component("comp1").physics(physics).feature(tag).selection().named(selection);
    model.component("comp1").physics(physics).feature(tag).set("PeriodicType", "Floquet");
    model.component("comp1").physics(physics).feature(tag)
        .set("kFloquet", new String[][]{{"kx"}, {"ky"}, {"0"}});
  }

  private static void zeroAuxiliarySolidConstitutive(Model model, String physics) {
    model.component("comp1").physics(physics).feature("lemm1").set("E_mat", "userdef");
    model.component("comp1").physics(physics).feature("lemm1").set("E", "0[Pa]");
    model.component("comp1").physics(physics).feature("lemm1").set("nu_mat", "userdef");
    model.component("comp1").physics(physics).feature("lemm1").set("nu", "0.3");
    model.component("comp1").physics(physics).feature("lemm1").set("rho_mat", "userdef");
    model.component("comp1").physics(physics).feature("lemm1").set("rho", "0[kg/m^3]");
  }

  private static void addIsotropicMaterial(
      Model model, String tag, String label, String selection,
      String young, String poisson, String density) {
    model.component("comp1").material().create(tag, "Common");
    model.component("comp1").material(tag).label(label);
    model.component("comp1").material(tag).selection().named(selection);
    model.component("comp1").material(tag).propertyGroup("def").set("density", density);
    model.component("comp1").material(tag).propertyGroup()
        .create("Enu", "Young's modulus and Poisson's ratio");
    model.component("comp1").material(tag).propertyGroup("Enu").set("E", young);
    model.component("comp1").material(tag).propertyGroup("Enu").set("nu", poisson);
  }

  private static void addPzt5hMaterial(
      Model model, String tag, String selection, double polarity) {
    model.component("comp1").material().create(tag, "Common");
    model.component("comp1").material(tag).label("PZT-5H " + tag);
    model.component("comp1").material(tag).selection().named(selection);
    model.component("comp1").material(tag).propertyGroup("def")
        .set("relpermittivity", new String[]{"1704.4", "0", "0", "0", "1704.4", "0", "0", "0", "1433.6"});
    model.component("comp1").material(tag).propertyGroup("def")
        .set("density", "7500[kg/m^3]");
    model.component("comp1").material(tag).propertyGroup()
        .create("StressCharge", "Stress-charge form");
    model.component("comp1").material(tag).propertyGroup("StressCharge")
        .set("cE", new String[]{"1.27205e+011[Pa]", "8.02122e+010[Pa]", "8.46702e+010[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "8.02122e+010[Pa]", "1.27205e+011[Pa]", "8.46702e+010[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "8.46702e+010[Pa]", "8.46702e+010[Pa]", "1.17436e+011[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "2.29885e+010[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "2.29885e+010[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "0[Pa]", "2.34742e+010[Pa]"});
    String e31 = Double.toString(polarity * -6.62281) + "[C/m^2]";
    String e33 = Double.toString(polarity * 23.2403) + "[C/m^2]";
    String e15 = Double.toString(polarity * 17.0345) + "[C/m^2]";
    model.component("comp1").material(tag).propertyGroup("StressCharge")
        .set("eES", new String[]{"0[C/m^2]", "0[C/m^2]", e31, "0[C/m^2]", "0[C/m^2]", e31, "0[C/m^2]", "0[C/m^2]", e33, "0[C/m^2]", e15, "0[C/m^2]", e15, "0[C/m^2]", "0[C/m^2]", "0[C/m^2]", "0[C/m^2]", "0[C/m^2]"});
    model.component("comp1").material(tag).propertyGroup("StressCharge")
        .set("epsilonrS", new String[]{"1704.4", "0", "0", "0", "1704.4", "0", "0", "0", "1433.6"});
  }

  private static void createCylinder(
      Model model, String tag, String radius, String height, String z,
      boolean resultSelection) {
    model.component("comp1").geom("geom1").create(tag, "Cylinder");
    model.component("comp1").geom("geom1").feature(tag).set("r", radius);
    model.component("comp1").geom("geom1").feature(tag).set("h", height);
    model.component("comp1").geom("geom1").feature(tag)
        .set("pos", new String[]{"0", "0", z});
    if (resultSelection) {
      model.component("comp1").geom("geom1").feature(tag).set("selresult", true);
    }
  }

  private static void createAnnulus(
      Model model, String tag, String outer, String inner, String height,
      String z, boolean resultSelection) {
    String outerTag = tag + "Outer";
    String innerTag = tag + "Inner";
    createCylinder(model, outerTag, outer, height, z, false);
    createCylinder(model, innerTag, inner, height, z, false);
    model.component("comp1").geom("geom1").create(tag, "Difference");
    model.component("comp1").geom("geom1").feature(tag)
        .selection("input").set(outerTag);
    model.component("comp1").geom("geom1").feature(tag)
        .selection("input2").set(innerTag);
    if (resultSelection) {
      model.component("comp1").geom("geom1").feature(tag).set("selresult", true);
    }
  }

  private static double[][] c4vNodeTable(double[] sectorRadii, double latticeUm) {
    if (sectorRadii.length != 4) {
      throw new IllegalArgumentException("Four radial nodes are required in 0--45 degrees");
    }
    double[][] table = new double[24][2];
    int[] radialIndex = new int[]{0, 1, 2, 3, 2, 1};
    for (int index = 0; index < table.length; index++) {
      int local = index % 6;
      double radius = latticeUm * sectorRadii[radialIndex[local]];
      double angle = index * Math.PI / 12.0;
      table[index][0] = radius * Math.cos(angle);
      table[index][1] = radius * Math.sin(angle);
    }
    return table;
  }

  private static void createPolygonPrism(
      Model model, String tag, double[][] vertices, String height, String z,
      boolean resultSelection) {
    String workplane = tag + "Wp";
    model.component("comp1").geom("geom1").create(workplane, "WorkPlane");
    model.component("comp1").geom("geom1").feature(workplane).set("quickplane", "xy");
    model.component("comp1").geom("geom1").feature(workplane).set("quickz", z);
    model.component("comp1").geom("geom1").feature(workplane).geom()
        .create("pol1", "Polygon");
    model.component("comp1").geom("geom1").feature(workplane).geom()
        .feature("pol1").set("source", "table");
    model.component("comp1").geom("geom1").feature(workplane).geom()
        .feature("pol1").set("table", vertices);
    model.component("comp1").geom("geom1").create(tag, "Extrude");
    model.component("comp1").geom("geom1").feature(tag)
        .selection("input").set(workplane);
    model.component("comp1").geom("geom1").feature(tag)
        .set("distance", new String[]{height});
    if (resultSelection) {
      model.component("comp1").geom("geom1").feature(tag).set("selresult", true);
    }
  }

  private static void createPolygonAnnulus(
      Model model, String tag, double[][] outerVertices, double[][] innerVertices,
      String height, String z, boolean resultSelection) {
    String outerTag = tag + "Outer";
    String innerTag = tag + "Inner";
    createPolygonPrism(model, outerTag, outerVertices, height, z, false);
    createPolygonPrism(model, innerTag, innerVertices, height, z, false);
    model.component("comp1").geom("geom1").create(tag, "Difference");
    model.component("comp1").geom("geom1").feature(tag)
        .selection("input").set(outerTag);
    model.component("comp1").geom("geom1").feature(tag)
        .selection("input2").set(innerTag);
    if (resultSelection) {
      model.component("comp1").geom("geom1").feature(tag).set("selresult", true);
    }
  }

  private static void createFaceBox(
      Model model, String tag, double xmin, double xmax, double ymin, double ymax,
      double zmin, double zmax) {
    model.component("comp1").selection().create(tag, "Box");
    model.component("comp1").selection(tag).geom("geom1", 2);
    model.component("comp1").selection(tag).set("entitydim", 2);
    model.component("comp1").selection(tag).set("condition", "inside");
    model.component("comp1").selection(tag).set("xmin", xmin);
    model.component("comp1").selection(tag).set("xmax", xmax);
    model.component("comp1").selection(tag).set("ymin", ymin);
    model.component("comp1").selection(tag).set("ymax", ymax);
    model.component("comp1").selection(tag).set("zmin", zmin);
    model.component("comp1").selection(tag).set("zmax", zmax);
  }

  private static void createUnion(
      Model model, String tag, String[] inputs, int entityDimension) {
    model.component("comp1").selection().create(tag, "Union");
    model.component("comp1").selection(tag).geom("geom1", entityDimension);
    model.component("comp1").selection(tag).set("entitydim", entityDimension);
    model.component("comp1").selection(tag).set("input", inputs);
  }

  public static void main(String[] args) {
    run();
  }
}
