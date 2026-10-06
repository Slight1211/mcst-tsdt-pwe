import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;

/** Same thick MCST model, joint mesh refinement and audited Gamma prescription. */
public class SolveA75JointPath {
  public static Model run() throws Exception {
    String input=System.getenv("MCST_JOINT_INPUT"),dir=System.getenv("MCST_JOINT_OUTPUT");
    if(input==null || dir==null || new File(dir+"/frequencies.csv").exists())throw new IllegalArgumentException("Invalid input/output");
    if(!new File(input).isFile())throw new FileNotFoundException("Missing MCST_JOINT_INPUT MPH checkpoint; historical checkpoints are omitted from the public release.");
    Model m=ModelUtil.load("JointA75",input);
    double a=m.param().evaluate("a"),h=m.param().evaluate("hp");
    if(Math.abs(a-75e-6)>1e-14 || Math.abs(h-24e-6)>1e-14)throw new IllegalArgumentException("Wrong geometry");
    if(Double.parseDouble(m.component("comp1").mesh("mesh1").feature("size1").getString("hauto"))!=2 ||
       Double.parseDouble(m.component("comp1").mesh("mesh1").feature("drySweep").feature("dis1").getString("numelem"))!=4)
      throw new IllegalArgumentException("Not M2L4 input");
    if(Double.parseDouble(m.sol("sol1").feature("e1").getString("neigs"))!=20 ||
       Math.abs(Double.parseDouble(m.sol("sol1").feature("e1").getString("rtol"))-1e-9)>1e-15 ||
       !"1[MHz]".equals(m.sol("sol1").feature("e1").getString("shift")))throw new IllegalArgumentException("Unexpected source eigensolver");
    System.out.println("INPUT_GEOMETRY a="+a+" h="+h+" mesh=2 layers=4 neigs=20");
    m.sol("sol1").clearSolutionData();
    m.component("comp1").mesh("mesh1").feature("size1").set("hauto",1);
    m.component("comp1").mesh("mesh1").feature("drySweep").feature("dis1").set("numelem",6);
    m.component("comp1").mesh("mesh1").run();
    System.out.println("JOINT_MESH_READY mesh="+(int)Double.parseDouble(m.component("comp1").mesh("mesh1").feature("size1").getString("hauto"))+
      " layers="+(int)Double.parseDouble(m.component("comp1").mesh("mesh1").feature("drySweep").feature("dis1").getString("numelem")));
    m.save(dir+"/meshed.mph");PrintWriter out=new PrintWriter(dir+"/frequencies.csv");
    try {
      out.println("point_index,kx_pi_over_a,ky_pi_over_a,mode_index,frequency_mhz,imag_frequency_mhz,vertical_fraction,core_fraction,core_vertical_fraction,solve_seconds");
      for(int p=0;p<13;p++) {
        double x=p<=4?p/4.0:(p<=8?1.0:(12-p)/4.0),y=p<=4?0.0:(p<=8?(p-4)/4.0:(12-p)/4.0);
        boolean gamma=p==0||p==12;String shift=gamma?"1[MHz]":"1.0E-4[MHz]";double rtol=gamma?1e-9:1e-6;
        m.param().set("kx",x+"*pi/a");m.param().set("ky",y+"*pi/a");
        m.sol("sol1").feature("e1").set("shift",shift);m.sol("sol1").feature("e1").set("rtol",rtol);m.sol("sol1").clearSolutionData();
        System.out.println("POINT_SETTINGS point="+p+" kx="+x+" ky="+y+" shift="+m.sol("sol1").feature("e1").getString("shift")+
          " rtol="+m.sol("sol1").feature("e1").getString("rtol")+" neigs="+(int)Double.parseDouble(m.sol("sol1").feature("e1").getString("neigs")));
        long start=System.nanoTime();m.sol("sol1").runAll();double seconds=(System.nanoTime()-start)/1e9;
        double[][] re=m.result().numerical("gevFreq").getReal(),im=m.result().numerical("gevFreq").getImag();
        if(re[0].length!=20)throw new IllegalArgumentException("Unexpected eigenpair count");
        for(int j=0;j<re[0].length;j++)out.println(p+","+x+","+y+","+(j+1)+","+re[0][j]+","+im[0][j]+","+re[1][j]+","+re[2][j]+","+re[3][j]+","+seconds);
        out.flush();m.save(dir+"/checkpoints/point_"+p+".mph");System.out.println("JOINT_POINT_COMPLETE point="+p+" seconds="+seconds);
      }
    } finally {out.close();}
    System.out.println("JOINT_PATH_COMPLETE");return m;
  }
  public static void main(String[] args)throws Exception{run();}
}
