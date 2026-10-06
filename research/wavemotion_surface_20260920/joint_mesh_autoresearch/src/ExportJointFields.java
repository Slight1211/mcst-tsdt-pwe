import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;

/** Read-only export on a shared physical midpoint grid; never calls a solver. */
public class ExportJointFields {
  public static Model run() throws Exception {
    String source=System.getenv("MCST_FIELD_MODEL"), output=System.getenv("MCST_FIELD_CSV");
    if(source==null || output==null)throw new IllegalArgumentException("Missing input/output");
    if(new File(output).exists())throw new IllegalArgumentException("Refuse to overwrite export");
    int n=Integer.parseInt(System.getenv("MCST_FIELD_NXY"));
    int nz=Integer.parseInt(System.getenv("MCST_FIELD_NZ"));
    if(n<2 || nz<2)throw new IllegalArgumentException("Insufficient sampling");
    if(!new File(source).isFile())throw new FileNotFoundException("Missing MCST_FIELD_MODEL MPH input; historical checkpoints are omitted from the public release.");
    Model m=ModelUtil.load("JointFields",source);
    for(String tag:m.sol("sol1").feature().tags()) {
      System.out.println("SAVED_SOLVER_FEATURE "+tag+" "+m.sol("sol1").feature(tag).getType());
      for(String property:m.sol("sol1").feature(tag).properties()) {
        if(property.toLowerCase().matches(".*(tol|shift|eig|trans|linp).*")) {
          try { System.out.println("SAVED_SOLVER_PROPERTY "+tag+" "+property+"="
              +m.sol("sol1").feature(tag).getString(property)); }
          catch(Exception ex) { System.out.println("SAVED_SOLVER_PROPERTY_NONSCALAR "+tag+" "+property); }
        }
      }
    }
    String geometryUnit=m.component("comp1").geom("geom1").lengthUnit();
    System.out.println("GEOMETRY_UNIT="+geometryUnit);
    String normalizedUnit=geometryUnit.replace('\u00b5','u').replace('\u03bc','u');
    if(!normalizedUnit.equals("um"))
      throw new IllegalArgumentException("Unexpected geometry coordinate unit: "+geometryUnit);
    double a=m.param().evaluate("a"), h=m.param().evaluate("hp"); // SI meters
    int count=n*n*nz;
    double[][] xyz=new double[3][count];
    for(int iz=0;iz<nz;iz++)for(int iy=0;iy<n;iy++)for(int ix=0;ix<n;ix++){
      int p=(iz*n+iy)*n+ix;
      xyz[0][p]=a*1e6*((ix+0.5)/n-0.5);
      xyz[1][p]=a*1e6*((iy+0.5)/n-0.5);
      xyz[2][p]=h*1e6*((iz+0.5)/nz-0.5);
    }
    m.result().numerical().create("jointFields","Interp");
    m.result().numerical("jointFields").set("data","dset1");
    m.result().numerical("jointFields").set("expr",new String[]{"u","v","w","x","y","z","solid.rho"});
    m.result().numerical("jointFields").set("unit",new String[]{"m","m","m","m","m","m","kg/m^3"});
    m.result().numerical("jointFields").selection().geom(3);
    m.result().numerical("jointFields").selection().all();
    m.result().numerical("jointFields").setInterpolationCoordinates(xyz);
    double[][][] re=m.result().numerical("jointFields").getData();
    double[][][] im=m.result().numerical("jointFields").getImagData();
    int nm=re[0].length;
    PrintWriter out=new PrintWriter(output);
    out.print("x_m,y_m,z_m,rho_kg_m3,volume_weight_m3");
    for(int mode=1;mode<=nm;mode++)for(String c:new String[]{"u","v","w"})
      out.print(","+c+mode+"_real,"+c+mode+"_imag");
    out.println();
    for(int p=0;p<count;p++){
      for(int c=0;c<3;c++)if(!Double.isFinite(re[c+3][0][p]) ||
         Math.abs(re[c+3][0][p]-xyz[c][p]*1e-6)>Math.max(a,h)*1e-8)
        throw new RuntimeException("Coordinate mismatch "+p);
      if(!(re[6][0][p]>0))throw new RuntimeException("Invalid density "+p);
      out.print(re[3][0][p]+","+re[4][0][p]+","+re[5][0][p]+","+re[6][0][p]+","+(a*a*h/count));
      for(int mode=0;mode<nm;mode++)for(int c=0;c<3;c++){
        if(!Double.isFinite(re[c][mode][p]) || !Double.isFinite(im[c][mode][p]))
          throw new RuntimeException("Nonfinite displacement");
        out.print(","+re[c][mode][p]+","+im[c][mode][p]);
      }
      out.println();
    }
    out.close();
    m.result().numerical().create("jointMeta","EvalGlobal");
    m.result().numerical("jointMeta").set("data","dset1");
    m.result().numerical("jointMeta").set("expr",new String[]{"freq","kx*a/pi","ky*a/pi"});
    m.result().numerical("jointMeta").set("unit",new String[]{"MHz","1","1"});
    double[][] mr=m.result().numerical("jointMeta").getReal(),mi=m.result().numerical("jointMeta").getImag();
    PrintWriter meta=new PrintWriter(output+".modes.csv");
    meta.println("mode,frequency_mhz,imag_frequency_mhz,kx_pi_over_a,ky_pi_over_a");
    for(int mode=0;mode<nm;mode++)meta.println((mode+1)+","+mr[0][mode]+","+mi[0][mode]+","+mr[1][mode]+","+mr[2][mode]);
    meta.close();
    ModelUtil.clear();
    System.out.println("FIELD_EXPORT_COMPLETE points="+count+" modes="+nm);
    return null;
  }
  public static void main(String[] args)throws Exception{run();}
}
