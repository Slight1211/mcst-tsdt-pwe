import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/** Portable publication adaptation of the completed homogeneous periodic-solid diagnostic.
 * Only output configuration is adapted; numerical algorithms and tolerances are unchanged.
 * Requires a licensed COMSOL installation; no historical model is loaded or modified.
 * MCST_BULK_OUTPUT_DIR must name an existing absolute directory; MCST_BULK_PREFIX
 * must be a filename stem containing only letters, digits, underscores, or hyphens.
 * Fixed 4^3/6^3 structured meshes; positive ell/a=0.05/0.10; six eigenpairs.
 * No model is saved, and existing output files are never overwritten.
 * The numerical tolerances below are pre-registered, not fitted to results.
 */
public class UniformMcstPeriodic {
  static final double A = 100e-6, E = 4.35e9, NU = 4.35/(2*1.59)-1, RHO = 1180.;
  static final double[] KAPI = {.18,.11,.07};

  static void selectionBox(Model m, String tag, int dim, int axis, double value) {
    m.component("comp1").selection().create(tag,"Box");
    m.component("comp1").selection(tag).geom("geom1",dim);
    m.component("comp1").selection(tag).set("entitydim",dim);
    m.component("comp1").selection(tag).set("condition","inside");
    String[] coordinate = {"x","y","z"};
    for(int j=0;j<3;j++) {
      m.component("comp1").selection(tag).set(coordinate[j]+"min",j==axis?value-.001:-.001);
      m.component("comp1").selection(tag).set(coordinate[j]+"max",j==axis?value+.001:100.001);
    }
  }

  static void periodic(Model m,String physics,String axis) {
    String tag="pc"+axis;
    m.component("comp1").physics(physics).create(tag,"PeriodicCondition",2);
    m.component("comp1").physics(physics).feature(tag).selection().named("pair"+axis);
    m.component("comp1").physics(physics).feature(tag).set("PeriodicType","Floquet");
    m.component("comp1").physics(physics).feature(tag).set("kFloquet",new String[][]{{"kx"},{"ky"},{"kz"}});
  }

  static void auxiliary(Model m,String tag,String[] fields,int order) {
    m.component("comp1").physics().create(tag,"SolidMechanics","geom1");
    m.component("comp1").physics(tag).field("displacement").component(fields);
    m.component("comp1").physics(tag).selection().all();
    m.component("comp1").physics(tag).prop("ShapeProperty").set("order_displacement",order);
    m.component("comp1").physics(tag).feature("lemm1").set("E_mat","userdef");
    m.component("comp1").physics(tag).feature("lemm1").set("E","0[Pa]");
    m.component("comp1").physics(tag).feature("lemm1").set("nu_mat","userdef");
    m.component("comp1").physics(tag).feature("lemm1").set("nu","0.3");
    m.component("comp1").physics(tag).feature("lemm1").set("rho_mat","userdef");
    m.component("comp1").physics(tag).feature("lemm1").set("rho","0[kg/m^3]");
  }

  static void writeNew(Path path,String text) throws Exception {
    BufferedWriter writer=Files.newBufferedWriter(path,StandardCharsets.UTF_8,
        StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE);
    try { writer.write(text); } finally { writer.close(); }
  }

  static double overlapSquared(double[][] re,double[][] im,int a,int b) {
    double na=0,nb=0,dr=0,di=0;
    for(int j=6;j<9;j++) {
      na+=re[j][a]*re[j][a]+im[j][a]*im[j][a];
      nb+=re[j][b]*re[j][b]+im[j][b]*im[j][b];
      dr+=re[j][a]*re[j][b]+im[j][a]*im[j][b];
      di+=re[j][a]*im[j][b]-im[j][a]*re[j][b];
    }
    return (dr*dr+di*di)/(na*nb);
  }

  static double projectionFraction(double[][] re,double[][] im,int j) {
    double amplitude2=0;
    for(int r=6;r<9;r++)amplitude2+=re[r][j]*re[r][j]+im[r][j]*im[r][j];
    return re[5][j]*amplitude2/re[3][j];
  }

  static String csv(String text) { return "\""+String.valueOf(text).replace("\"","\"\"")+"\""; }

  static void recordDiscretization(Model m,StringBuilder audit,String physics,int expectedOrder) {
    int order=Integer.parseInt(m.component("comp1").physics(physics).prop("ShapeProperty").getString("order_displacement"));
    if(order!=expectedOrder)throw new IllegalStateException("Unexpected displacement order for "+physics+": "+order);
    for(String property:m.component("comp1").physics(physics).prop("ShapeProperty").properties()) {
      String type=m.component("comp1").physics(physics).prop("ShapeProperty").getValueType(property);
      String value;
      try { value=m.component("comp1").physics(physics).prop("ShapeProperty").getString(property); }
      catch(Exception textUnavailable) {
        try { value=Arrays.toString(m.component("comp1").physics(physics).prop("ShapeProperty").getStringArray(property)); }
        catch(Exception arrayUnavailable) { value="UNREADABLE_WITH_SCALAR_OR_STRING_ARRAY_GETTERS"; }
      }
      audit.append(csv(physics)).append(',').append(csv(property)).append(',').append(csv(type)).append(',').append(csv(value)).append('\n');
    }
    System.out.println("BULK_DISCRETIZATION "+physics+" actual_order="+order+" shape_type_retained_at_COMSOL_default");
  }

  static boolean recordSelection(Model m,StringBuilder audit,String tag,int dimension,int expected) {
    int count=m.component("comp1").selection(tag).entities(dimension).length;
    boolean pass=count==expected;
    audit.append(tag).append(',').append(dimension).append(',').append(count).append(',').append(expected).append(',').append(pass).append('\n');
    System.out.println("BULK_SELECTION "+tag+" actual="+count+" expected="+expected);
    return pass;
  }

  static int elementDimension(String type) {
    if(type.equals("hex")||type.equals("tet")||type.equals("prism")||type.equals("pyr"))return 3;
    if(type.equals("tri")||type.equals("quad"))return 2;
    if(type.equals("edg"))return 1;
    if(type.equals("vtx"))return 0;
    return -1;
  }

  public static Model run() throws Exception {
    int n=Integer.parseInt(System.getenv("MCST_BULK_N"));
    double ratio=Double.parseDouble(System.getenv("MCST_BULK_RATIO"));
    String outputDirText=System.getenv("MCST_BULK_OUTPUT_DIR");
    String prefix=System.getenv("MCST_BULK_PREFIX");
    if((n!=4&&n!=6)||(ratio!=.05&&ratio!=.10))throw new IllegalArgumentException("Frozen mesh/length choices only");
    if(outputDirText==null||outputDirText.trim().isEmpty())
      throw new IllegalArgumentException("MCST_BULK_OUTPUT_DIR must name an existing absolute directory");
    Path outputDir=Paths.get(outputDirText);
    if(!outputDir.isAbsolute())throw new IllegalArgumentException("MCST_BULK_OUTPUT_DIR must be absolute");
    outputDir=outputDir.toAbsolutePath().normalize();
    if(!Files.isDirectory(outputDir))throw new IllegalArgumentException("MCST_BULK_OUTPUT_DIR must already exist as a directory");
    if(prefix==null||!prefix.matches("[A-Za-z0-9_-]+"))
      throw new IllegalArgumentException("MCST_BULK_PREFIX must match [A-Za-z0-9_-]+ and must not contain a path");
    Path stem=outputDir.resolve(prefix).toAbsolutePath().normalize();
    if(!stem.getParent().equals(outputDir))throw new IllegalArgumentException("Output stem outside MCST_BULK_OUTPUT_DIR");
    Path rawPath=Paths.get(stem.toString()+".raw.csv"),selectedPath=Paths.get(stem.toString()+".selected.csv"),checkPath=Paths.get(stem.toString()+".checks.csv");
    Path meshPath=Paths.get(stem.toString()+".mesh.csv"),selectionPath=Paths.get(stem.toString()+".selection.csv");
    Path discretizationPath=Paths.get(stem.toString()+".discretization.csv");
    for(Path p:new Path[]{rawPath,selectedPath,checkPath,meshPath,selectionPath,discretizationPath})if(Files.exists(p))throw new IllegalArgumentException("Refuse overwrite "+p);
    double mu=E/(2*(1+NU)),lambda=E*NU/((1+NU)*(1-2*NU));
    double k2=0;for(double v:KAPI)k2+=Math.pow(Math.PI*v/A,2);
    double ell=ratio*A,omegaT2=mu/RHO*(k2+ell*ell*k2*k2/4),omegaL2=(lambda+2*mu)/RHO*k2;
    double fT=Math.sqrt(omegaT2)/(2*Math.PI),fL=Math.sqrt(omegaL2)/(2*Math.PI),omegaTClass2=mu/RHO*k2;
    final double freqTol=n==4?1e-4:2.5e-5;
    final double incrementTol=n==4?.05:.02;
    final double amplitudeGramConditionTol=2.;
    final double minimumG0Projection=n==4?.99:.995;
    final double polarizationTol=n==4?.01:.005;
    final double constraintTol=n==4?.02:.01;
    final double longitudinalScaleTol=n==4?.01:.005;
    final double splitTol=n==4?2e-4:5e-5;
    System.out.println("BULK_PREREG n="+n+" ell/a="+ratio+" analytic_fT_Hz="+fT+" analytic_fL_Hz="+fL);
    System.out.println("BULK_SCOPE homogeneous fully periodic solid; not an interface/free-face validation");
    Model m=null;
    try {
      m=ModelUtil.create("UniformMcstPeriodic");
      m.param().set("a","100[um]");m.param().set("Ebulk","4.35[GPa]");
      m.param().set("nubulk","4.35/(2*1.59)-1");m.param().set("rhobulk","1180[kg/m^3]");
      m.param().set("mubulk","Ebulk/(2*(1+nubulk))");m.param().set("ellMcst",Double.toString(ratio)+"*a");
      m.param().set("kx","0.18*pi/a");m.param().set("ky","0.11*pi/a");m.param().set("kz","0.07*pi/a");
      m.param().set("knorm","sqrt(kx^2+ky^2+kz^2)");
      m.component().create("comp1",true);m.component("comp1").sorder("quadratic");
      m.component("comp1").geom().create("geom1",3);m.component("comp1").geom("geom1").lengthUnit("um");
      m.component("comp1").geom("geom1").create("block1","Block");
      m.component("comp1").geom("geom1").feature("block1").set("size",new String[]{"a","a","a"});
      m.component("comp1").geom("geom1").run();
      String[] axes={"x","y","z"};
      for(int j=0;j<3;j++) {
        selectionBox(m,"lo"+axes[j],2,j,0);selectionBox(m,"hi"+axes[j],2,j,100);
        m.component("comp1").selection().create("pair"+axes[j],"Union");
        m.component("comp1").selection("pair"+axes[j]).geom("geom1",2);
        m.component("comp1").selection("pair"+axes[j]).set("entitydim",2);
        m.component("comp1").selection("pair"+axes[j]).set("input",new String[]{"lo"+axes[j],"hi"+axes[j]});
      }
      selectionBox(m,"bottomEdges",1,2,0);
      StringBuilder selectionAudit=new StringBuilder("selection,dimension,actual_count,expected_count,passed\n");
      boolean selectionsPass=true;
      for(String axis:axes) {
        selectionsPass=recordSelection(m,selectionAudit,"lo"+axis,2,1)&&selectionsPass;
        selectionsPass=recordSelection(m,selectionAudit,"hi"+axis,2,1)&&selectionsPass;
        selectionsPass=recordSelection(m,selectionAudit,"pair"+axis,2,2)&&selectionsPass;
      }
      selectionsPass=recordSelection(m,selectionAudit,"bottomEdges",1,4)&&selectionsPass;
      writeNew(selectionPath,selectionAudit.toString());
      if(!selectionsPass)throw new IllegalStateException("Periodic-face/mesh-edge selection counts failed before physics assembly");
      m.component("comp1").material().create("mat1","Common");
      m.component("comp1").material("mat1").selection().all();
      m.component("comp1").material("mat1").propertyGroup("def").set("density","rhobulk");
      m.component("comp1").material("mat1").propertyGroup().create("Enu","Young's modulus and Poisson's ratio");
      m.component("comp1").material("mat1").propertyGroup("Enu").set("E","Ebulk");
      m.component("comp1").material("mat1").propertyGroup("Enu").set("nu","nubulk");
      m.component("comp1").physics().create("solid","SolidMechanics","geom1");
      m.component("comp1").physics("solid").prop("ShapeProperty").set("order_displacement",2);
      auxiliary(m,"mcstRot",new String[]{"mrx","mry","mrz"},2);
      auxiliary(m,"mcstLag",new String[]{"mlx","mly","mlz"},1);
      StringBuilder discretizationAudit=new StringBuilder("physics,shape_property,value_type,actual_value\n");
      recordDiscretization(m,discretizationAudit,"solid",2);
      recordDiscretization(m,discretizationAudit,"mcstRot",2);
      recordDiscretization(m,discretizationAudit,"mcstLag",1);
      writeNew(discretizationPath,discretizationAudit.toString());
      for(String physics:new String[]{"solid","mcstRot","mcstLag"})for(String axis:axes)periodic(m,physics,axis);
      String chi2="d(mrx,x)^2+d(mry,y)^2+d(mrz,z)^2+2*(0.5*(d(mrx,y)+d(mry,x)))^2"
        +"+2*(0.5*(d(mrx,z)+d(mrz,x)))^2+2*(0.5*(d(mry,z)+d(mrz,y)))^2";
      String energy="mubulk*("+chi2+")+mubulk/ellMcst^2*(mlx*mrx+mly*mry+mlz*mrz)"
        +"-mubulk/(2*ellMcst)*(mlx*(d(w,y)-d(v,z))+mly*(d(u,z)-d(w,x))+mlz*(d(v,x)-d(u,y)))";
      m.component("comp1").physics("mcstRot").create("mcstWeak","WeakContribution",3);
      m.component("comp1").physics("mcstRot").feature("mcstWeak").selection().all();
      m.component("comp1").physics("mcstRot").feature("mcstWeak").set("weakExpression","-test("+energy+")");
      m.component("comp1").physics("mcstRot").feature("mcstWeak").set("integrationOrder",8);
      m.component("comp1").mesh().create("mesh1");
      m.component("comp1").mesh("mesh1").create("map1","Map");
      m.component("comp1").mesh("mesh1").feature("map1").selection().named("loz");
      m.component("comp1").mesh("mesh1").feature("map1").create("dis1","Distribution");
      m.component("comp1").mesh("mesh1").feature("map1").feature("dis1").selection().named("bottomEdges");
      m.component("comp1").mesh("mesh1").feature("map1").feature("dis1").set("numelem",n);
      m.component("comp1").mesh("mesh1").create("sweep1","Sweep");
      m.component("comp1").mesh("mesh1").feature("sweep1").selection().geom("geom1",3);
      m.component("comp1").mesh("mesh1").feature("sweep1").selection().all();
      m.component("comp1").mesh("mesh1").feature("sweep1").selection("sourceface").named("loz");
      m.component("comp1").mesh("mesh1").feature("sweep1").selection("targetface").named("hiz");
      m.component("comp1").mesh("mesh1").feature("sweep1").create("dis1","Distribution");
      m.component("comp1").mesh("mesh1").feature("sweep1").feature("dis1").set("numelem",n);
      m.component("comp1").mesh("mesh1").run();
      int actualHex=0,otherVolume=0;boolean knownMeshTypes=true;
      StringBuilder meshAudit=new StringBuilder("element_type,topological_dimension,actual_count\n");
      for(String type:m.component("comp1").mesh("mesh1").getTypes()) {
        int count=m.component("comp1").mesh("mesh1").getNumElem(type),dimension=elementDimension(type);
        meshAudit.append(type).append(',').append(dimension).append(',').append(count).append('\n');
        System.out.println("BULK_MESH type="+type+" dimension="+dimension+" actual_count="+count);
        if(type.equals("hex"))actualHex+=count;
        else if(dimension==3)otherVolume+=count;
        if(dimension<0&&count>0)knownMeshTypes=false;
      }
      writeNew(meshPath,meshAudit.toString());
      if(actualHex!=n*n*n||otherVolume!=0||!knownMeshTypes)
        throw new IllegalStateException("Measured mesh is not the frozen all-hexahedral grid; no eigensolve started");
      m.component("comp1").cpl().create("intAll","Integration");
      m.component("comp1").cpl("intAll").selection().all();
      m.component("comp1").cpl("intAll").set("intorder","8");
      m.study().create("std1");m.study("std1").create("eig","Eigenfrequency");
      m.study("std1").feature("eig").set("neigs",6);m.study("std1").feature("eig").set("neigsactive",true);
      m.study("std1").feature("eig").set("shift","0.1[MHz]");m.study("std1").feature("eig").set("shiftactive",true);
      m.study("std1").createAutoSequences("all");
      m.sol("sol1").feature("e1").set("rtol",1e-9);
      long start=System.nanoTime();m.sol("sol1").runAll();double seconds=(System.nanoTime()-start)/1e9;
      int solutionVectorLength=m.sol("sol1").getU().length;
      String theta2="0.25*(abs(d(w,y)-d(v,z))^2+abs(d(u,z)-d(w,x))^2+abs(d(v,x)-d(u,y))^2)";
      String residual2="abs(mrx/ellMcst-0.5*(d(w,y)-d(v,z)))^2+abs(mry/ellMcst-0.5*(d(u,z)-d(w,x)))^2"
        +"+abs(mrz/ellMcst-0.5*(d(v,x)-d(u,y)))^2";
      // COMSOL: u_dest=exp(-i*k_F.(r_dest-r_src))*u_src.
      // Therefore the G=0 Fourier amplitude uses exp(+i*k_F.x).
      String unorm="abs(u)^2+abs(v)^2+abs(w)^2",phase="exp(i*(kx*x+ky*y+kz*z))";
      m.result().numerical().create("checkOnly","EvalGlobal");m.result().numerical("checkOnly").set("data","dset1");
      m.result().numerical("checkOnly").set("expr",new String[]{"freq","intAll("+residual2+")","intAll("+theta2+")",
        "intAll("+unorm+")","intAll(abs((kx*u+ky*v+kz*w)/knorm)^2)/intAll("+unorm+")","intAll(1)",
        "intAll(u*"+phase+")/a^3","intAll(v*"+phase+")/a^3","intAll(w*"+phase+")/a^3"});
      m.result().numerical("checkOnly").set("unit",new String[]{"Hz","m^3","m^3","m^5","1","m^3","m","m","m"});
      double[][] re=m.result().numerical("checkOnly").getReal(),im=m.result().numerical("checkOnly").getImag();
      StringBuilder raw=new StringBuilder("n,ell_over_a,raw_mode,frequency_hz,imag_frequency_hz,residual_norm_squared_m3,rotation_norm_squared_m3,displacement_norm_squared_m5,longitudinal_fraction,volume_m3,g0_projection_fraction,mean_u_real_m,mean_u_imag_m,mean_v_real_m,mean_v_imag_m,mean_w_real_m,mean_w_imag_m,solution_vector_length,solve_seconds\n");
      List<Integer> transverse=new ArrayList<Integer>(),longitudinal=new ArrayList<Integer>();
      for(int j=0;j<re[0].length;j++) {
        raw.append(n).append(',').append(ratio).append(',').append(j+1);
        raw.append(',').append(re[0][j]).append(',').append(im[0][j]);
        for(int r=1;r<6;r++)raw.append(',').append(re[r][j]);
        raw.append(',').append(projectionFraction(re,im,j));
        for(int r=6;r<9;r++)raw.append(',').append(re[r][j]).append(',').append(im[r][j]);
        raw.append(',').append(solutionVectorLength).append(',').append(seconds).append('\n');
        if(Double.isFinite(re[0][j])&&Math.abs(re[0][j]/fT-1)<.02&&re[4][j]<.05)transverse.add(j);
        if(Double.isFinite(re[0][j])&&Math.abs(re[0][j]/fL-1)<.02&&re[4][j]>.95)longitudinal.add(j);
      }
      writeNew(rawPath,raw.toString());
      if(transverse.size()!=2||longitudinal.size()!=1)
        throw new IllegalStateException("Expected two transverse and one longitudinal G=0 mode; raw outputs retained");
      int t1=transverse.get(0),t2=transverse.get(1),l=longitudinal.get(0);
      double overlap=overlapSquared(re,im,t1,t2),split=Math.abs(re[0][t1]-re[0][t2])/fT;
      // Degenerate eigenvectors are not assumed to be solver-orthogonal.  The
      // normalized amplitude Gram eigenvalues are 1 +/- sqrt(overlapSquared).
      double coherence=Math.sqrt(overlap),gramCondition=(1+coherence)/(1-coherence);
      boolean allPass=Double.isFinite(gramCondition)&&gramCondition>=1&&gramCondition<=amplitudeGramConditionTol&&split<=splitTol;
      StringBuilder selected=new StringBuilder("n,ell_over_a,type,raw_mode,frequency_hz,analytic_frequency_hz,relative_frequency_error,mcst_increment_relative_error,longitudinal_fraction,constraint_error_rotation_normalized,constraint_error_k_u_normalized,rotation_over_k_u,relative_imaginary_frequency,g0_projection_fraction,passed\n");
      int[] chosen={t1,t2,l};String[] names={"T1","T2","L"};
      for(int c=0;c<3;c++) {
        int j=chosen[c];boolean isT=c<2;double exact=isT?fT:fL;
        double ferror=Math.abs(re[0][j]/exact-1),imaginary=Math.abs(im[0][j]/re[0][j]);
        double ecKU=Math.sqrt(re[1][j]/(k2*re[3][j])),thetaKU=Math.sqrt(re[2][j]/(k2*re[3][j]));
        double ecRot=isT?Math.sqrt(re[1][j]/re[2][j]):Double.NaN;
        double projection=projectionFraction(re,im,j);
        double increment=isT?Math.abs((Math.pow(2*Math.PI*re[0][j],2)-omegaTClass2)/(omegaT2-omegaTClass2)-1):Double.NaN;
        boolean pass=Double.isFinite(ferror)&&ferror<=freqTol&&imaginary<=1e-8&&re[3][j]>0&&re[1][j]>=0&&re[2][j]>=0
          &&Math.abs(re[5][j]/(A*A*A)-1)<1e-10&&Double.isFinite(projection)&&projection>=minimumG0Projection&&projection<=1+1e-8;
        if(isT)pass=pass&&re[4][j]<=polarizationTol&&Double.isFinite(ecRot)&&ecRot<=constraintTol&&increment<=incrementTol;
        else pass=pass&&re[4][j]>=1-polarizationTol&&Double.isFinite(ecKU)&&ecKU<=longitudinalScaleTol&&thetaKU<=longitudinalScaleTol;
        allPass=allPass&&pass;
        selected.append(n).append(',').append(ratio).append(',').append(names[c]).append(',').append(j+1).append(',')
          .append(re[0][j]).append(',').append(exact).append(',').append(ferror).append(',').append(isT?Double.toString(increment):"").append(',')
          .append(re[4][j]).append(',').append(isT?Double.toString(ecRot):"").append(',').append(ecKU).append(',').append(thetaKU)
          .append(',').append(imaginary).append(',').append(projection).append(',').append(pass).append('\n');
      }
      writeNew(selectedPath,selected.toString());
      writeNew(checkPath,"n,ell_over_a,expected_hex_elements,actual_hex_elements,other_volume_elements,solution_vector_length,integration_order,frequency_tolerance,mcst_increment_tolerance,polarization_tolerance,transverse_constraint_tolerance,longitudinal_scaled_tolerance,transverse_split,transverse_split_tolerance,transverse_amplitude_overlap_squared,transverse_amplitude_gram_condition,transverse_amplitude_gram_condition_tolerance,minimum_g0_projection_fraction,passed\n"
        +n+","+ratio+","+(n*n*n)+","+actualHex+","+otherVolume+","+solutionVectorLength+",8,"+freqTol+","+incrementTol+","+polarizationTol+","+constraintTol+","+longitudinalScaleTol+","+split+","+splitTol+","+overlap+","+gramCondition+","+amplitudeGramConditionTol+","+minimumG0Projection+","+allPass+"\n");
      System.out.println("BULK_BENCHMARK_COMPLETE passed="+allPass+" dof_vector_length="+solutionVectorLength+" seconds="+seconds);
      if(!allPass)throw new IllegalStateException("Pre-registered acceptance failed; raw diagnostic outputs retained without retuning");
    } finally { ModelUtil.clear(); }
    return null;
  }
  public static void main(String[] args)throws Exception{run();}
}
